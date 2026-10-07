//! Error-state EKF for BENCH_VI (EuRoC replay): f32, no_std, alloc-free.
//!
//! Nominal state (stored): p(3) world pos, v(3) world vel, q(4) body->world
//! ([w,x,y,z]), b_a(3), b_g(3). Error state (covariance): dp,dv,dth,dba,dbg
//! with dth body-frame (R_true = R*(I+[dth]x)), so H for a pose fix is ±I.
//! P is full 15x15 (900 B, keep in internal SRAM, never PSRAM).
//! IMU predict runs at IMU rate; correct_pose runs per VO fix (~0.2 Hz).
//! Conventions: specific force a_m = R^T*(a_world - g) + b_a, g = [0,0,-9.81].

const N: usize = 15;

fn sqrt_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x.is_infinite() {
        return x;
    }
    let mut r = f32::from_bits((x.to_bits() + 0x3F80_0000) >> 1);
    for _ in 0..3 {
        r = 0.5 * (r + x / r);
    }
    r
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[allow(dead_code)] // used by tests / kept for the quaternion algebra
#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn skew(v: [f32; 3]) -> [[f32; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

#[inline]
fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

// ---- quaternions ([w,x,y,z], body -> world) ----

#[inline]
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

#[inline]
fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn quat_norm(q: [f32; 4]) -> [f32; 4] {
    let n = sqrt_f32(dot3([q[1], q[2], q[3]], [q[1], q[2], q[3]]) + q[0] * q[0]);
    if !(n > 0.0) || !n.is_finite() {
        return [1.0, 0.0, 0.0, 0.0];
    }
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

#[allow(dead_code)] // used by tests / kept for the quaternion algebra
#[inline]
fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qv = [q[1], q[2], q[3]];
    let t = cross3(qv, v);
    let t2 = [2.0 * t[0], 2.0 * t[1], 2.0 * t[2]];
    let c = cross3(qv, t2);
    [v[0] + q[0] * t2[0] + c[0], v[1] + q[0] * t2[1] + c[1], v[2] + q[0] * t2[2] + c[2]]
}

fn rot_mat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// Rotation matrix -> [w,x,y,z] quat (Shepperd branches, sqrt only, no libm).
pub fn quat_from_mat(m: [[f32; 3]; 3]) -> [f32; 4] {
    let tr = m[0][0] + m[1][1] + m[2][2];
    let q = if tr > 0.0 {
        let s = sqrt_f32(tr + 1.0) * 2.0;
        [(0.25 * s), (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = sqrt_f32(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    } else if m[1][1] > m[2][2] {
        let s = sqrt_f32(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    } else {
        let s = sqrt_f32(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]
    };
    quat_norm(q)
}
/// Body-frame attitude residual of measurement vs prediction (2x vector
/// part of the error quat; ≈ angle-axis for small angles). Exposed for the
/// harness-side residual-feedback observer.
pub fn attitude_residual(q_pred: [f32; 4], q_meas: [f32; 4]) -> [f32; 3] {    let qm = quat_norm(q_meas);
    let mut qe = quat_mul(quat_conj(q_pred), qm);
    if qe[0] < 0.0 {
        qe = [-qe[0], -qe[1], -qe[2], -qe[3]];
    }
    [2.0 * qe[1], 2.0 * qe[2], 2.0 * qe[3]]
}

/// Right-multiply a body-frame small-rotation correction (disturbance
/// observer use): q' = q ⊗ quat(dphi).
pub fn quat_correct_body(q: [f32; 4], dphi: [f32; 3]) -> [f32; 4] {
    quat_norm(quat_mul(q, quat_from_rotvec(dphi)))
}
/// |w*dt| << 1 steps predict takes at 200 Hz; normalizes after anyway.
/// Small-angle rotvec -> quat via Taylor (no libm). Accurate for the
/// |w*dt| << 1 steps predict takes at 200 Hz; normalizes after anyway.
pub fn quat_from_rotvec(phi: [f32; 3]) -> [f32; 4] {
    let t2 = dot3(phi, phi);
    let s = 0.5 - t2 / 48.0;
    quat_norm([1.0 - t2 / 8.0, phi[0] * s, phi[1] * s, phi[2] * s])
}

// ---- 6x6 Cholesky (factor once, solve many) ----

fn chol6_factor(s: [[f32; 6]; 6]) -> Option<[[f32; 6]; 6]> {
    let mut l = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut v = s[i][j];
            for k in 0..j {
                v -= l[i][k] * l[j][k];
            }
            if i == j {
                if !(v > 0.0) || !v.is_finite() {
                    return None;
                }
                l[i][i] = sqrt_f32(v);
            } else {
                l[i][j] = v / l[j][j];
            }
        }
    }
    Some(l)
}

fn chol6_solve(l: [[f32; 6]; 6], b: [f32; 6]) -> Option<[f32; 6]> {
    let mut y = [0.0; 6];
    for i in 0..6 {
        let mut v = b[i];
        for k in 0..i {
            v -= l[i][k] * y[k];
        }
        y[i] = v / l[i][i];
    }
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        let mut v = y[i];
        for k in (i + 1)..6 {
            v -= l[k][i] * x[k];
        }
        x[i] = v / l[i][i];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

fn chol3_factor(s: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let mut l = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut v = s[i][j];
            for k in 0..j {
                v -= l[i][k] * l[j][k];
            }
            if i == j {
                if !(v > 0.0) || !v.is_finite() {
                    return None;
                }
                l[i][i] = sqrt_f32(v);
            } else {
                l[i][j] = v / l[j][j];
            }
        }
    }
    Some(l)
}

fn chol3_solve(l: [[f32; 3]; 3], b: [f32; 3]) -> Option<[f32; 3]> {
    let mut y = [0.0; 3];
    for i in 0..3 {
        let mut v = b[i];
        for k in 0..i {
            v -= l[i][k] * y[k];
        }
        y[i] = v / l[i][i];
    }
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        let mut v = y[i];
        for k in (i + 1)..3 {
            v -= l[k][i] * x[k];
        }
        x[i] = v / l[i][i];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

// ---- filter ----

/// Continuous-time noise std devs (per sqrt(Hz)).
#[derive(Clone, Copy)]
pub struct EkfNoise {
    pub sigma_a: f32,  // m/s^2
    pub sigma_g: f32,  // rad/s
    pub sigma_ba: f32, // m/s^2 bias walk
    pub sigma_bg: f32, // rad/s bias walk
}

impl Default for EkfNoise {
    /// EuRoC VI-Sensor (ADIS16448) values from mav0/imu0/sensor.yaml
    /// (noise densities + random walks, per sqrt(Hz)).
    fn default() -> Self {
        Self { sigma_a: 2.0e-3, sigma_g: 1.6968e-4, sigma_ba: 3.0e-3, sigma_bg: 1.9393e-5 }
    }
}

#[derive(Clone)]
pub struct Ekf {
    pub p: [f32; 3],
    pub v: [f32; 3],
    pub q: [f32; 4],
    pub ba: [f32; 3],
    pub bg: [f32; 3],
    pub g: [f32; 3],
    pub p_cov: [[f32; N]; N],
    pub noise: EkfNoise,
}

impl Ekf {
    /// p0[i] = initial std of group i (dp, dv, dth, dba, dbg).
    pub fn new(p: [f32; 3], q: [f32; 4], noise: EkfNoise, p0: [f32; 5]) -> Self {
        let mut c = [[0.0; N]; N];
        for g in 0..5 {
            for k in 0..3 {
                c[g * 3 + k][g * 3 + k] = p0[g] * p0[g];
            }
        }
        Self {
            p,
            v: [0.0; 3],
            q: quat_norm(q),
            ba: [0.0; 3],
            bg: [0.0; 3],
            g: [0.0, 0.0, -9.81],
            p_cov: c,
            noise,
        }
    }

    fn symmetrize(&mut self) {
        for i in 0..N {
            for j in (i + 1)..N {
                let m = 0.5 * (self.p_cov[i][j] + self.p_cov[j][i]);
                self.p_cov[i][j] = m;
                self.p_cov[j][i] = m;
            }
        }
    }

    /// Body->world rotation at the current nominal attitude.
    pub fn body_to_world(&self) -> [[f32; 3]; 3] {
        rot_mat(self.q)
    }

    /// Nominal-only integration of one IMU sample (µs, no covariance).
    /// Returns the bias-corrected body (accel, gyro) for covariance batching,
    /// or None (no state change) on bad inputs.
    pub fn propagate_nominal(
        &mut self,
        a_m: [f32; 3],
        w_m: [f32; 3],
        dt: f32,
    ) -> Option<([f32; 3], [f32; 3])> {
        if !(dt > 0.0) || !dt.is_finite() {
            return None;
        }
        if !a_m.iter().chain(w_m.iter()).all(|v| v.is_finite()) {
            return None;
        }
        let w = [w_m[0] - self.bg[0], w_m[1] - self.bg[1], w_m[2] - self.bg[2]];
        let a = [a_m[0] - self.ba[0], a_m[1] - self.ba[1], a_m[2] - self.ba[2]];
        self.q = quat_norm(quat_mul(self.q, quat_from_rotvec([w[0] * dt, w[1] * dt, w[2] * dt])));
        let r = rot_mat(self.q);
        let aw = [
            r[0][0] * a[0] + r[0][1] * a[1] + r[0][2] * a[2] + self.g[0],
            r[1][0] * a[0] + r[1][1] * a[1] + r[1][2] * a[2] + self.g[1],
            r[2][0] * a[0] + r[2][1] * a[1] + r[2][2] * a[2] + self.g[2],
        ];
        self.v = [self.v[0] + aw[0] * dt, self.v[1] + aw[1] * dt, self.v[2] + aw[2] * dt];
        self.p = [self.p[0] + self.v[0] * dt, self.p[1] + self.v[1] * dt, self.p[2] + self.v[2] * dt];
        Some((a, w))
    }

    /// Covariance-only propagation over dt, linearized at reference corrected
    /// body accel/gyro (a_ref, w_ref) and rotation r. Exact F*P*F' + Qd with
    /// F's static sparsity written out (54 nonzeros); ~4x the dense cost.
    /// Pair with per-sample propagate_nominal: average the refs over a batch.
    pub fn propagate_covariance(
        &mut self,
        a_ref: [f32; 3],
        w_ref: [f32; 3],
        r: [[f32; 3]; 3],
        dt: f32,
    ) {
        if !(dt > 0.0) || !dt.is_finite() {
            return;
        }
        if !a_ref.iter().chain(w_ref.iter()).all(|v| v.is_finite()) {
            return;
        }
        let m = mat3_mul(r, skew(a_ref)); // -M feeds dv wrt dth
        let ws = skew(w_ref);
        // FP = F*P, one F-row pattern at a time.
        let mut fp = [[0.0; N]; N];
        for j in 0..N {
            for i in 0..3 {
                fp[i][j] = self.p_cov[i][j] + dt * self.p_cov[3 + i][j];
                fp[9 + i][j] = self.p_cov[9 + i][j];
                fp[12 + i][j] = self.p_cov[12 + i][j];
            }
            for i in 0..3 {
                let mut s = self.p_cov[3 + i][j];
                for k in 0..3 {
                    s += -m[i][k] * dt * self.p_cov[6 + k][j]
                        - r[i][k] * dt * self.p_cov[9 + k][j];
                }
                fp[3 + i][j] = s;
                let mut t = self.p_cov[6 + i][j];
                for k in 0..3 {
                    t += -ws[i][k] * dt * self.p_cov[6 + k][j];
                }
                fp[6 + i][j] = t - dt * self.p_cov[12 + i][j];
            }
        }
        // P = FP*F' + Qd, same row patterns transposed.
        let n = self.noise;
        let qd = [0.0, n.sigma_a * n.sigma_a * dt, n.sigma_g * n.sigma_g * dt,
                  n.sigma_ba * n.sigma_ba * dt, n.sigma_bg * n.sigma_bg * dt];
        for i in 0..N {
            for j in 0..3 {
                self.p_cov[i][j] = fp[i][j] + dt * fp[i][3 + j];
            }
            for j in 3..6 {
                let jj = j - 3;
                let mut s = fp[i][j];
                for k in 0..3 {
                    s += fp[i][6 + k] * (-m[jj][k] * dt) + fp[i][9 + k] * (-r[jj][k] * dt);
                }
                self.p_cov[i][j] = s;
            }
            for j in 6..9 {
                let jj = j - 6;
                let mut s = fp[i][j];
                for k in 0..3 {
                    s += fp[i][6 + k] * (-ws[jj][k] * dt);
                }
                self.p_cov[i][j] = s + fp[i][12 + jj] * (-dt);
            }
            for j in 9..N {
                self.p_cov[i][j] = fp[i][j];
            }
            let g = i / 3; // group of row i (0=dp has no process noise)
            if g >= 1 {
                self.p_cov[i][i] += qd[g];
            }
        }
        self.symmetrize();
    }

    /// One IMU sample, nominal + covariance. a_m = specific force, dt in s.
    pub fn predict(&mut self, a_m: [f32; 3], w_m: [f32; 3], dt: f32) {
        if let Some((a, w)) = self.propagate_nominal(a_m, w_m, dt) {
            let r = rot_mat(self.q);
            self.propagate_covariance(a, w, r, dt);
        }
    }

    /// Fuse one VO pose fix. r_pos/r_att = isotropic meas variances (m^2, rad^2).
    /// Returns residual norm, or None if rejected (bad inputs / S not PD).
    pub fn correct_pose(
        &mut self,
        p_meas: [f32; 3],
        q_meas: [f32; 4],
        r_pos: f32,
        r_att: f32,
    ) -> Option<f32> {
        if !(r_pos > 0.0) || !(r_att > 0.0) || !r_pos.is_finite() || !r_att.is_finite() {
            return None;
        }
        if !p_meas.iter().all(|v| v.is_finite()) {
            return None;
        }
        let qm = quat_norm(q_meas);
        let mut qe = quat_mul(quat_conj(self.q), qm);
        if qe[0] < 0.0 {
            qe = [-qe[0], -qe[1], -qe[2], -qe[3]];
        }
        let y = [
            p_meas[0] - self.p[0],
            p_meas[1] - self.p[1],
            p_meas[2] - self.p[2],
            2.0 * qe[1],
            2.0 * qe[2],
            2.0 * qe[3],
        ];
        if !y.iter().all(|v| v.is_finite()) {
            return None;
        }
        // S = H*P*H' + R; H picks rows dp(0..3) and dth(6..9).
        let idx = [0, 1, 2, 6, 7, 8];
        let mut s = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                s[i][j] = self.p_cov[idx[i]][idx[j]];
            }
            s[i][i] += if i < 3 { r_pos } else { r_att };
        }
        let l = chol6_factor(s)?;
        let z = chol6_solve(l, y)?;
        // PHt (15x6), dx = PHt*z, K rows via S^-1.
        let mut pht = [[0.0; 6]; N];
        for i in 0..N {
            for k in 0..6 {
                pht[i][k] = self.p_cov[i][idx[k]];
            }
        }
        let mut dx = [0.0; N];
        for i in 0..N {
            dx[i] = pht[i][0] * z[0] + pht[i][1] * z[1] + pht[i][2] * z[2]
                + pht[i][3] * z[3] + pht[i][4] * z[4] + pht[i][5] * z[5];
        }
        if !dx.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut k = [[0.0; 6]; N];
        for i in 0..N {
            k[i] = chol6_solve(l, pht[i])?;
        }
        // Inject (dth is body-frame -> right-multiply).
        for i in 0..3 {
            self.p[i] += dx[i];
            self.v[i] += dx[3 + i];
            self.ba[i] += dx[9 + i];
            self.bg[i] += dx[12 + i];
        }
        self.q = quat_norm(quat_mul(self.q, quat_from_rotvec([dx[6], dx[7], dx[8]])));
        // Joseph update: P = A*P*A' + K*R*K', A = I - K*H.
        let mut a = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                a[i][j] = if i == j { 1.0 } else { 0.0 };
            }
            for kk in 0..3 {
                a[i][kk] -= k[i][kk];
                a[i][6 + kk] -= k[i][3 + kk];
            }
        }
        let mut ap = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut s = 0.0;
                for kk in 0..N {
                    s += a[i][kk] * self.p_cov[kk][j];
                }
                ap[i][j] = s;
            }
        }
        for i in 0..N {
            for j in 0..N {
                let mut s = 0.0;
                for kk in 0..N {
                    s += ap[i][kk] * a[j][kk];
                }
                // K*R*K' with isotropic R (r_pos on k<3, r_att on k>=3).
                for kk in 0..6 {
                    s += k[i][kk] * k[j][kk] * if kk < 3 { r_pos } else { r_att };
                }
                self.p_cov[i][j] = s;
            }
        }
        self.symmetrize();
        Some(sqrt_f32(dot3([y[0], y[1], y[2]], [y[0], y[1], y[2]])))
    }

    /// Fuse a world-velocity pseudo-measurement (e.g. differenced VO
    /// positions / dt). H picks dv rows (3..6). Returns residual norm.
    pub fn correct_velocity(&mut self, v_meas: [f32; 3], r_vel: f32) -> Option<f32> {
        if !(r_vel > 0.0) || !r_vel.is_finite() {
            return None;
        }
        if !v_meas.iter().all(|v| v.is_finite()) {
            return None;
        }
        let y = [v_meas[0] - self.v[0], v_meas[1] - self.v[1], v_meas[2] - self.v[2]];
        if !y.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut s = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] = self.p_cov[3 + i][3 + j];
            }
            s[i][i] += r_vel;
        }
        let l = chol3_factor(s)?;
        let z = chol3_solve(l, y)?;
        let mut pht = [[0.0; 3]; N];
        for i in 0..N {
            for k in 0..3 {
                pht[i][k] = self.p_cov[i][3 + k];
            }
        }
        let mut dx = [0.0; N];
        for i in 0..N {
            dx[i] = pht[i][0] * z[0] + pht[i][1] * z[1] + pht[i][2] * z[2];
        }
        if !dx.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut k = [[0.0; 3]; N];
        for i in 0..N {
            k[i] = chol3_solve(l, pht[i])?;
        }
        for i in 0..3 {
            self.p[i] += dx[i];
            self.v[i] += dx[3 + i];
            self.ba[i] += dx[9 + i];
            self.bg[i] += dx[12 + i];
        }
        self.q = quat_norm(quat_mul(self.q, quat_from_rotvec([dx[6], dx[7], dx[8]])));
        let mut a = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                a[i][j] = if i == j { 1.0 } else { 0.0 };
            }
            for kk in 0..3 {
                a[i][3 + kk] -= k[i][kk];
            }
        }
        let mut ap = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut s = 0.0;
                for kk in 0..N {
                    s += a[i][kk] * self.p_cov[kk][j];
                }
                ap[i][j] = s;
            }
        }
        for i in 0..N {
            for j in 0..N {
                let mut s = 0.0;
                for kk in 0..N {
                    s += ap[i][kk] * a[j][kk];
                }
                for kk in 0..3 {
                    s += k[i][kk] * k[j][kk] * r_vel;
                }
                self.p_cov[i][j] = s;
            }
        }
        self.symmetrize();
        Some(sqrt_f32(dot3(y, y)))
    }

    /// Fuse a world-position measurement only (no attitude): H picks dp
    /// rows (0..3). For testing whether VO attitude fixes are the poison.
    pub fn correct_position(&mut self, p_meas: [f32; 3], r_pos: f32) -> Option<f32> {
        if !(r_pos > 0.0) || !r_pos.is_finite() {
            return None;
        }
        if !p_meas.iter().all(|v| v.is_finite()) {
            return None;
        }
        let y = [p_meas[0] - self.p[0], p_meas[1] - self.p[1], p_meas[2] - self.p[2]];
        if !y.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut s = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] = self.p_cov[i][j];
            }
            s[i][i] += r_pos;
        }
        let l = chol3_factor(s)?;
        let z = chol3_solve(l, y)?;
        let mut pht = [[0.0; 3]; N];
        for i in 0..N {
            for k in 0..3 {
                pht[i][k] = self.p_cov[i][k];
            }
        }
        let mut dx = [0.0; N];
        for i in 0..N {
            dx[i] = pht[i][0] * z[0] + pht[i][1] * z[1] + pht[i][2] * z[2];
        }
        if !dx.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut k = [[0.0; 3]; N];
        for i in 0..N {
            k[i] = chol3_solve(l, pht[i])?;
        }
        for i in 0..3 {
            self.p[i] += dx[i];
            self.v[i] += dx[3 + i];
            self.ba[i] += dx[9 + i];
            self.bg[i] += dx[12 + i];
        }
        self.q = quat_norm(quat_mul(self.q, quat_from_rotvec([dx[6], dx[7], dx[8]])));
        let mut a = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                a[i][j] = if i == j { 1.0 } else { 0.0 };
            }
            for kk in 0..3 {
                a[i][kk] -= k[i][kk];
            }
        }
        let mut ap = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut s = 0.0;
                for kk in 0..N {
                    s += a[i][kk] * self.p_cov[kk][j];
                }
                ap[i][j] = s;
            }
        }
        for i in 0..N {
            for j in 0..N {
                let mut s = 0.0;
                for kk in 0..N {
                    s += ap[i][kk] * a[j][kk];
                }
                for kk in 0..3 {
                    s += k[i][kk] * k[j][kk] * r_pos;
                }
                self.p_cov[i][j] = s;
            }
        }
        self.symmetrize();
        Some(sqrt_f32(dot3(y, y)))
    }

    pub fn pos_std(&self) -> [f32; 3] {
        [sqrt_f32(self.p_cov[0][0]), sqrt_f32(self.p_cov[1][1]), sqrt_f32(self.p_cov[2][2])]
    }

    /// Attitude std (rad) from the dth block (indices 6..9).
    pub fn att_std(&self) -> [f32; 3] {
        [sqrt_f32(self.p_cov[6][6]), sqrt_f32(self.p_cov[7][7]), sqrt_f32(self.p_cov[8][8])]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn quat_from_mat_roundtrip() {
        let c = 0.7071068;
        let m = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]; // 90 deg z
        let q = quat_from_mat(m);
        assert!(approx(q[0].abs(), c, 1e-4) && approx(q[3].abs(), c, 1e-4));
        let r = rot_mat(q);
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx(r[i][j], m[i][j], 1e-4));
            }
        }
        let qi = quat_from_mat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert!(approx(qi[0], 1.0, 1e-5));
    }

    #[test]
    fn quat_rotate_90z() {
        let c = 0.7071068;
        let q = quat_norm([c, 0.0, 0.0, c]); // 90 deg about z
        let v = quat_rotate(q, [1.0, 0.0, 0.0]);
        assert!(approx(v[0], 0.0, 1e-4) && approx(v[1], 1.0, 1e-4) && approx(v[2], 0.0, 1e-4));
    }

    #[test]
    fn stationary_holds_pose() {
        let mut e = Ekf::new([0.0; 3], [1.0, 0.0, 0.0, 0.0], EkfNoise::default(),
            [0.1, 0.1, 0.01, 0.01, 0.001]);
        for _ in 0..200 {
            e.predict([0.0, 0.0, 9.81], [0.0; 3], 0.005); // specific force cancels g
        }
        assert!(e.p.iter().all(|v| approx(*v, 0.0, 1e-3)));
        assert!(e.v.iter().all(|v| approx(*v, 0.0, 1e-2)));
        let v = quat_rotate(e.q, [1.0, 0.0, 0.0]);
        assert!(approx(v[0], 1.0, 1e-3) && approx(v[1], 0.0, 1e-3));
    }

    #[test]
    fn constant_accel_integrates() {
        let mut e = Ekf::new([0.0; 3], [1.0, 0.0, 0.0, 0.0], EkfNoise::default(),
            [0.1, 0.1, 0.01, 0.01, 0.001]);
        for _ in 0..200 {
            e.predict([1.0, 0.0, 9.81], [0.0; 3], 0.005); // 1 m/s^2 in +x
        }
        assert!(approx(e.v[0], 1.0, 0.02));
        assert!(approx(e.p[0], 0.5, 0.03));
    }

    #[test]
    fn gyro_yaw_90deg() {
        let mut e = Ekf::new([0.0; 3], [1.0, 0.0, 0.0, 0.0], EkfNoise::default(),
            [0.1, 0.1, 0.01, 0.01, 0.001]);
        let w = core::f32::consts::FRAC_PI_2; // 90 deg/s for 1 s
        for _ in 0..200 {
            e.predict([0.0, 0.0, 9.81], [0.0, 0.0, w], 0.005);
        }
        let v = quat_rotate(e.q, [1.0, 0.0, 0.0]);
        assert!(approx(v[0], 0.0, 0.02) && approx(v[1], 1.0, 0.02));
    }

    #[test]
    fn correction_pulls_position() {
        let mut e = Ekf::new([0.0; 3], [1.0, 0.0, 0.0, 0.0], EkfNoise::default(),
            [1.0, 0.1, 0.01, 0.01, 0.001]);
        e.p = [1.0, 0.5, -0.3]; // drifted
        let r = e.correct_pose([0.0; 3], [1.0, 0.0, 0.0, 0.0], 0.01, 0.001);
        assert!(r.is_some());
        assert!(e.p.iter().all(|v| v.abs() < 0.05));
        assert!(e.pos_std().iter().all(|v| *v < 0.5)); // covariance shrank
    }

    #[test]
    fn correction_pulls_yaw() {
        let mut e = Ekf::new([0.0; 3], [1.0, 0.0, 0.0, 0.0], EkfNoise::default(),
            [0.1, 0.1, 0.2, 0.01, 0.001]);
        // Yaw the estimate ~5 deg off.
        for _ in 0..20 {
            e.predict([0.0, 0.0, 9.81], [0.0, 0.0, 0.087], 0.005);
        }
        let r = e.correct_pose([0.0; 3], [1.0, 0.0, 0.0, 0.0], 0.01, 1e-6);
        assert!(r.is_some());
        let v = quat_rotate(e.q, [1.0, 0.0, 0.0]);
        assert!(approx(v[0], 1.0, 0.01) && v[1].abs() < 0.01);
    }

    #[test]
    fn attitude_residual_small_yaw() {
        let r = attitude_residual([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
        assert!(r.iter().all(|v| v.abs() < 1e-6));
        // 0.1 rad yaw: residual z should read ~0.1.
        let qm = quat_norm([1.0, 0.0, 0.0, 0.05]);
        let r = attitude_residual([1.0, 0.0, 0.0, 0.0], qm);
        assert!((r[2] - 0.1).abs() < 0.002 && r[0].abs() < 1e-6 && r[1].abs() < 1e-6);
    }

    #[test]
    fn position_only_leaves_attitude_measurement_out() {
        // correct_position must move p without needing/using any attitude
        // measurement: yaw the state, fix position, yaw must survive.
        let mut e = Ekf::new([0.0; 3], [1.0, 0.0, 0.0, 0.0], EkfNoise::default(),
            [1.0, 0.1, 0.01, 0.01, 0.001]);
        e.p = [1.0, 0.0, 0.0];
        let r = e.correct_position([0.0; 3], 0.01);
        assert!(r.is_some());
        assert!(e.p[0].abs() < 0.05);
        let v = quat_rotate(e.q, [1.0, 0.0, 0.0]);
        assert!((v[0] - 1.0).abs() < 1e-4 && v[1].abs() < 1e-4);
    }

    #[test]
    fn velocity_fix_updates_v_directly() {
        // No propagation: P is diagonal so correct_pose leaves v alone,
        // but correct_velocity must move it (H picks dv rows).
        let mut e = Ekf::new([0.0; 3], [1.0, 0.0, 0.0, 0.0], EkfNoise::default(),
            [0.1, 0.1, 0.01, 0.01, 0.001]);
        e.v = [2.0, 0.0, 0.0]; // wrong, no motion history
        let r = e.correct_velocity([0.0; 3], 0.01);
        assert!(r.is_some());
        assert!(e.v[0] < 1.9, "v not pulled: {:?}", e.v);
        assert!(e.pos_std().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn split_rate_matches_combined() {
        // 10 samples: per-sample predict vs nominal-per-sample + one
        // batch covariance step. Means must match closely, P approximately.
        let mk = || Ekf::new([0.1, -0.2, 0.05], [1.0, 0.0, 0.0, 0.0],
            EkfNoise::default(), [0.1, 0.1, 0.01, 0.01, 0.001]);
        let mut full = mk();
        let mut split = mk();
        let (mut aa, mut ww) = ([0.0; 3], [0.0; 3]);
        for i in 0..10 {
            let a = [0.3 + 0.01 * i as f32, 0.0, 9.81];
            let w = [0.0, 0.0, 0.05 + 0.001 * i as f32];
            full.predict(a, w, 0.005);
            let (ac, wc) = split.propagate_nominal(a, w, 0.005).unwrap();
            for k in 0..3 {
                aa[k] += ac[k];
                ww[k] += wc[k];
            }
        }
        for k in 0..3 {
            aa[k] /= 10.0;
            ww[k] /= 10.0;
        }
        split.propagate_covariance(aa, ww, split.body_to_world(), 0.05);
        for k in 0..3 {
            assert!((full.p[k] - split.p[k]).abs() < 1e-4, "p[{k}] drift");
            assert!((full.v[k] - split.v[k]).abs() < 1e-4, "v[{k}] drift");
        }
        for k in 0..15 {
            let d = (full.p_cov[k][k] - split.p_cov[k][k]).abs()
                / full.p_cov[k][k].max(1e-9);
            assert!(d < 0.05, "P[{k}] rel diff {d}");
        }
    }

    #[test]
    fn nominal_rejects_without_mutating() {
        let mut e = Ekf::new([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0],
            EkfNoise::default(), [0.1, 0.1, 0.01, 0.01, 0.001]);
        assert!(e.propagate_nominal([0.0, 0.0, 9.81], [0.0; 3], -0.005).is_none());
        assert!(e.propagate_nominal([f32::NAN, 0.0, 9.81], [0.0; 3], 0.005).is_none());
        assert!(e.p == [1.0, 2.0, 3.0]);
        let (a, w) = e.propagate_nominal([0.0, 0.0, 9.81], [0.1, 0.0, 0.0], 0.005).unwrap();
        assert!(a == [0.0, 0.0, 9.81] && w[0] == 0.1); // zero biases
    }

    #[test]
    fn rejects_bad_inputs() {
        let mut e = Ekf::new([0.0; 3], [1.0, 0.0, 0.0, 0.0], EkfNoise::default(),
            [0.1, 0.1, 0.01, 0.01, 0.001]);
        assert!(e.correct_pose([0.0; 3], [1.0, 0.0, 0.0, 0.0], -1.0, 0.001).is_none());
        assert!(e.correct_pose([f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0.01, 0.001).is_none());
        e.predict([0.0, 0.0, 9.81], [0.0; 3], -0.005); // ignored, stays finite
        assert!(e.p.iter().chain(e.v.iter()).all(|v| v.is_finite()));
    }
}
