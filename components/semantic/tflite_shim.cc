// Implementation of the C shim declared in tflite_shim.h. Only one model is
// ever loaded per boot, so the resolver/interpreter live in function-local
// statics and the model/arena pointers are held for the lifetime of the app.
#include "tflite_shim.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
// Generous op set covering yolov8n (conv/dw-conv, SiLU, SPPF maxpool,
// nearest upsample, DFL softmax/split, concat, transpose, reshape, quantize).
constexpr int kNumOps = 50;

tflite::MicroInterpreter *g_interp = nullptr;
TfLiteTensor *g_input = nullptr;
const char *g_err = "not loaded";

void register_ops(tflite::MicroMutableOpResolver<kNumOps> &r) {
  // Return values ignored on purpose: a missing op surfaces later as an
  // AllocateTensors failure naming the op.
  r.AddConv2D();
  r.AddDepthwiseConv2D();
  r.AddFullyConnected();
  r.AddAdd();
  r.AddSub();
  r.AddMul();
  r.AddDiv();
  r.AddLogistic();
  r.AddSoftmax();
  r.AddExp();
  r.AddMaxPool2D();
  r.AddAveragePool2D();
  r.AddResizeNearestNeighbor();
  r.AddResizeBilinear();
  r.AddConcatenation();
  r.AddSplit();
  r.AddSplitV();
  r.AddPack();
  r.AddUnpack();
  r.AddTranspose();
  r.AddReshape();
  r.AddSqueeze();
  r.AddExpandDims();
  r.AddPad();
  r.AddPadV2();
  r.AddStridedSlice();
  r.AddSlice();
  r.AddShape();
  r.AddGather();
  r.AddCast();
  r.AddQuantize();
  r.AddDequantize();
  r.AddMaximum();
  r.AddMinimum();
  r.AddReduceMax();
  r.AddRelu();
  r.AddRelu6();
  r.AddLeakyRelu();
  r.AddFloor();
  r.AddCeil();
  r.AddRound();
  r.AddNeg();
  r.AddAbs();
  r.AddSquare();
  r.AddSqrt();
  r.AddRsqrt();
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
  static tflite::MicroInterpreter interp(m, resolver, arena, arena_len);
  if (interp.AllocateTensors() != kTfLiteOk) {
    g_err = "AllocateTensors failed (missing op or arena too small?)";
    return -3;
  }
  g_input = interp.input(0);
  if (g_input == nullptr || interp.outputs_size() < 2) {
    g_err = "model needs 1 input and 2 outputs (boxes, scores)";
    return -4;
  }
  g_interp = &interp;
  g_err = "";
  return 0;
}

size_t semantic_input_size(void) { return g_input ? g_input->bytes : 0; }
int8_t *semantic_input_data(void) {
  return g_input ? g_input->data.int8 : nullptr;
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

int8_t *semantic_output_data(size_t i) {
  TfLiteTensor *t = output_at(i);
  return t ? t->data.int8 : nullptr;
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
  if (g_interp->Invoke() != kTfLiteOk) {
    g_err = "Invoke failed";
    return -2;
  }
  return 0;
}

const char *semantic_last_error(void) { return g_err; }
