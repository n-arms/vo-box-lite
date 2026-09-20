// Implementation of the C shim declared in tflite_shim.h. Only one model is
// ever loaded per boot, so the resolver/interpreter live in function-local
// statics and the model/arena pointers are held for the lifetime of the app.
#include "tflite_shim.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <stdio.h>
#include <string.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
// calc8 (CALC encoder) op set: quantize/pad/conv/pool/transpose plus the
// SHAPE->STRIDED_SLICE->PACK->RESHAPE flatten block.
constexpr int kNumOps = 12;

// MicroInterpreter hides the TfLiteEvalTensor array that kernels read/write;
// expose it via the protected context so a tensor's backing store can be swapped.
class SemanticInterpreter : public tflite::MicroInterpreter {
 public:
  SemanticInterpreter(const tflite::Model *model,
                      const tflite::MicroOpResolver &resolver, uint8_t *arena,
                      size_t arena_len, tflite::MicroProfilerInterface *profiler)
      : tflite::MicroInterpreter(model, resolver, arena, arena_len, nullptr,
                                 profiler) {}
  TfLiteEvalTensor *EvalTensor(int idx) {
    const TfLiteContext &ctx = this->context();
    return ctx.GetEvalTensor(&ctx, idx);
  }
};

SemanticInterpreter *g_interp = nullptr;
TfLiteTensor *g_input = nullptr;
const tflite::Model *g_model = nullptr;
const char *g_err = "not loaded";

// conv2's 4x4 filter is copied to internal RAM once (see
// pin_conv2_filter_sram), so esp-nn streams the weights from there.
bool g_conv2_pin_done = false;

// Minimal per-op profiler over esp_timer us (tags are TFLM's static op names).
class OpProfiler : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char *tag) override {
    if (!enabled_) return 0;
    cur_tag_ = tag;
    cur_start_ = esp_timer_get_time();
    return 1;
  }
  void EndEvent(uint32_t) override {
    if (!enabled_) return;
    ProfileEntry *e = find(cur_tag_, true);
    if (e) {
      e->us += esp_timer_get_time() - cur_start_;
      e->count++;
    }
  }
  void enable(int on) {
    enabled_ = (on != 0);
    if (!enabled_) reset();
  }
  void log_and_reset() {
    for (int i = 0; i < n_entries_; i++) {
      printf("semantic profile: %-16s %6d ms x%u\n", entries_[i].tag,
             (int)(entries_[i].us / 1000), (unsigned)entries_[i].count);
    }
    reset();
  }

 private:
  struct ProfileEntry {
    const char *tag = nullptr;
    int64_t us = 0;
    uint32_t count = 0;
  };
  static constexpr int kMaxProfileTags = 16;
  bool enabled_ = false;
  const char *cur_tag_ = nullptr;
  int64_t cur_start_ = 0;
  int n_entries_ = 0;
  ProfileEntry entries_[kMaxProfileTags];

  ProfileEntry *find(const char *tag, bool create) {
    for (int i = 0; i < n_entries_; i++) {
      if (entries_[i].tag == tag) return &entries_[i];
    }
    if (create && n_entries_ < kMaxProfileTags) {
      ProfileEntry *e = &entries_[n_entries_++];
      e->tag = tag;
      return e;
    }
    return nullptr;
  }
  void reset() {
    n_entries_ = 0;
    for (int i = 0; i < kMaxProfileTags; i++) {
      entries_[i] = ProfileEntry{};
    }
  }
};
OpProfiler g_profiler;

void register_ops(tflite::MicroMutableOpResolver<kNumOps> &r) {
  // Return values ignored on purpose: a missing op surfaces later as an
  // AllocateTensors failure naming the op.
  r.AddQuantize();
  r.AddPad();
  r.AddConv2D();
  r.AddMaxPool2D();
  r.AddTranspose();
  r.AddShape();
  r.AddStridedSlice();
  r.AddPack();
  r.AddReshape();
}
}  // namespace

int semantic_model_load(const uint8_t *model, size_t model_len, uint8_t *arena,
                        size_t arena_len) {
  (void)model_len;  // flatbuffer is self-describing
  if (g_interp != nullptr) {
    g_err = "already loaded";
    return -1;
  }
  const tflite::Model *m = tflite::GetModel(model);
  if (m->version() != TFLITE_SCHEMA_VERSION) {
    g_err = "model schema version mismatch";
    return -2;
  }

  static tflite::MicroMutableOpResolver<kNumOps> resolver;
  register_ops(resolver);
  static SemanticInterpreter interp(m, resolver, arena, arena_len, &g_profiler);
  if (interp.AllocateTensors() != kTfLiteOk) {
    g_err = "AllocateTensors failed (missing op or arena too small?)";
    return -3;
  }
  g_input = interp.input(0);
  if (g_input == nullptr || interp.outputs_size() < 1) {
    g_err = "model needs 1 input and >=1 output";
    return -4;
  }
  g_interp = &interp;
  g_model = m;
  g_err = "";
  return 0;
}

// Copy conv2's 4x4x64x128 filter into internal RAM and repoint its eval tensor,
// so esp-nn streams the weights from there (and skips its alignment copy).
// Called once from the first invoke; on failure the interpreter is left unchanged.
static bool pin_conv2_filter_sram(void) {
  if (g_interp == nullptr || g_model == nullptr) return false;
  g_conv2_pin_done = true;  // attempt once; SRAM won't grow later

  const tflite::SubGraph *sg = g_model->subgraphs()->Get(0);
  if (sg == nullptr) return false;
  int filt_idx = -1;
  for (size_t n = 0; n < sg->operators()->size() && filt_idx < 0; n++) {
    const tflite::Operator *op = sg->operators()->Get(n);
    const tflite::OperatorCode *oc =
        g_model->operator_codes()->Get(op->opcode_index());
    if (oc->builtin_code() != tflite::BuiltinOperator_CONV_2D) continue;
    const tflite::Tensor *ft = sg->tensors()->Get(op->inputs()->Get(1));
    const flatbuffers::Vector<int32_t> *fs = ft->shape();
    // calc8's only 4x4 kernel is conv2 ([128,4,4,64]).
    if (fs != nullptr && fs->size() == 4 && fs->Get(1) == 4 &&
        fs->Get(2) == 4) {
      filt_idx = op->inputs()->Get(1);
    }
  }
  if (filt_idx < 0) {
    g_err = "conv2 (4x4) not found";
    return false;
  }

  const flatbuffers::Vector<int32_t> *fs =
      sg->tensors()->Get(filt_idx)->shape();
  size_t filter_bytes = 1;
  for (size_t i = 0; i < fs->size(); i++) filter_bytes *= fs->Get(i);

  TfLiteEvalTensor *fe = g_interp->EvalTensor(filt_idx);
  uint8_t *buf = (uint8_t *)heap_caps_aligned_alloc(
      16, filter_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (fe == nullptr || fe->data.raw == nullptr || buf == nullptr) {
    if (buf != nullptr) heap_caps_free(buf);
    g_err = "conv2 filter SRAM pin failed";
    printf(
        "semantic sram: conv2 filter NOT pinned - need %u B, internal free "
        "%u B (largest %u B)\n",
        (unsigned)filter_bytes,
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL |
                                                   MALLOC_CAP_8BIT));
    return false;
  }
  memcpy(buf, fe->data.raw, filter_bytes);
  fe->data.raw = (char *)buf;
  g_err = "";
  printf(
      "semantic sram: conv2 filter pinned - %u B in DRAM (internal free "
      "%u B)\n",
      (unsigned)filter_bytes,
      (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
  return true;
}

size_t semantic_input_size(void) { return g_input ? g_input->bytes : 0; }
uint8_t *semantic_input_data(void) {
  return g_input ? g_input->data.uint8 : nullptr;
}

size_t semantic_output_count(void) {
  return g_interp ? g_interp->outputs_size() : 0;
}

static TfLiteTensor *output_at(size_t i) {
  if (g_interp == nullptr || i >= g_interp->outputs_size()) return nullptr;
  return g_interp->output(i);
}

size_t semantic_output_size(size_t i) {
  TfLiteTensor *t = output_at(i);
  return t ? t->bytes : 0;
}

uint8_t *semantic_output_data(size_t i) {
  TfLiteTensor *t = output_at(i);
  return t ? t->data.uint8 : nullptr;
}

float semantic_output_scale(size_t i) {
  TfLiteTensor *t = output_at(i);
  return t ? t->params.scale : 0.0f;
}

int32_t semantic_output_zero_point(size_t i) {
  TfLiteTensor *t = output_at(i);
  return t ? t->params.zero_point : 0;
}

int semantic_invoke(void) {
  if (g_interp == nullptr) {
    g_err = "no model loaded";
    return -1;
  }
  // First invoke (after camera/WiFi): pin conv2's weights to SRAM if they fit.
  // Failure is non-fatal (the kernel keeps reading the flash-mapped filter).
  if (!g_conv2_pin_done) pin_conv2_filter_sram();
  if (g_interp->Invoke() != kTfLiteOk) {
    g_err = "Invoke failed";
    return -2;
  }
  return 0;
}

const char *semantic_last_error(void) { return g_err; }

void semantic_profile_enable(int on) { g_profiler.enable(on); }
void semantic_profile_log(void) { g_profiler.log_and_reset(); }
