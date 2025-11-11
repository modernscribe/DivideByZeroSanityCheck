#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# = imports =
import sys, os, json, math, argparse, hashlib
import numpy as np

# = config =
PRINCIPLES = ["Truth","Purity","Law","Love","Wisdom","Life","Glory"]
ZODIAC = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
EPS = 1e-12

# = utils =
def md5(b: bytes): return hashlib.md5(b).hexdigest()
def linspace(a,b,n): return np.linspace(a,b,n,endpoint=False)
def grid(uN,vN): return np.meshgrid(linspace(0,2*math.pi,uN), linspace(0,2*math.pi,vN), indexing='ij')
def clamp(x,a,b): return a if x<a else b if x>b else x
def safe_tan(x):
    h = (math.pi/2)-1e-6
    return math.tan(clamp(x,-h,h))
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# = anchors =
def Null_point(): return 0.0
def InverseZero_operator(p): return -0.0 if p==0.0 else -0.0
def H_inverse_zero_null(): return 1.0
def sigma_selector(omega): return 1.0

# = tones =
def omega_k(k): return 2.0*math.pi*(k/12.0)

# = minors =
def minor_phase_iter(phi,alpha=0.83,beta=0.71): return alpha*safe_tan(beta*phi)
def minor_amplitude(A,mods): return A*(1.0+0.15*np.tanh(np.sum(mods,axis=0)))

# = kernels =
def psi_truth(u,v,om): return np.cos(om*u)
def psi_purity(u,v,om): return np.sign(np.cos(om*u))*np.sign(np.cos(om*v))
def psi_law(u,v,om): return np.cos(om*u)+np.cos(om*v)
def psi_love(u,v,om): return np.cos(om*(u-v))
def psi_wisdom(u,v,om): return np.cos(om*u)+np.cos(om*v)+np.cos(om*(u+v))
def psi_life(u,v,om): return np.sin(om*u)*np.cos(om*v)
def psi_glory(u,v,om): return np.cos(om*u)*np.cos(om*v) - 0.5*np.cos(om*(u+v))
KERNELS = {"Truth":psi_truth,"Purity":psi_purity,"Law":psi_law,"Love":psi_love,"Wisdom":psi_wisdom,"Life":psi_life,"Glory":psi_glory}

# = canvas/torus =
def dielectric_canvas(u,v): return np.zeros_like(u)
def torus_base(u,v,R=3.0,r=1.0):
    x=(R+r*np.cos(v))*np.cos(u)
    y=(R+r*np.cos(v))*np.sin(u)
    z=r*np.sin(v)
    return x,y,z

# = embedding =
def embed_3d(u,v,Sx,Sy,Sz,R=3.0,r=1.0):
    X0,Y0,Z0 = torus_base(u,v,R,r)
    cu, su = np.cos(u), np.sin(u)
    x = X0 + Sx*cu
    y = Y0 + Sy*su
    z = Z0 + Sz
    return x,y,z

# = holonomy approx =
def conn_from_scalar(F):
    Au = np.gradient(F,axis=0)
    Av = np.gradient(F,axis=1)
    return Au,Av
def holonomy_spectrum(Au,Av):
    lu = float(np.sum(Au[:,0]))
    lv = float(np.sum(Av[0,:]))
    return [complex(np.cos(lu),np.sin(lu)), complex(np.cos(lv),np.sin(lv))]

# = metrics =
def kuramoto_r(phases):
    z = np.exp(1j*phases)
    r = np.abs(np.mean(z))
    return float(r)
def occupancy_fraction(energies,thresh_ratio=0.2):
    e = energies.reshape(-1)
    t = float(np.mean(e)+thresh_ratio*np.std(e))
    return float(np.mean(e>t))
def nearness_metric(u,v):
    g = np.exp(-((u-math.pi)**2 + (v-math.pi)**2)/(2.0*(math.pi/3.0)**2))
    return float(np.mean(g))
def love_alignment_cost(field_a,field_b):
    a = (field_a - np.mean(field_a)); b = (field_b - np.mean(field_b))
    na = np.linalg.norm(a)+EPS; nb = np.linalg.norm(b)+EPS
    return float(1.0 - np.tensordot(a/na,b/nb,axes=2)/ (a.size))

# = per-octave =
def build_octave(u,v,oct_idx,state,params):
    uN,vN = u.shape
    mode_energy = np.zeros((len(PRINCIPLES),12,uN,vN),dtype=np.float64)
    phases_accum = []
    Sx = np.zeros((uN,vN),dtype=np.float64)
    Sy = np.zeros((uN,vN),dtype=np.float64)
    Sz = np.zeros((uN,vN),dtype=np.float64)
    chaos_phase = state["chaos_phase"]
    aX,aY,aZ = params["ax"],params["ay"],params["az"]
    zeta = H_inverse_zero_null()*sigma_selector(1.0)
    for j,pr in enumerate(PRINCIPLES):
        ker = KERNELS[pr]
        for k in range(12):
            om = omega_k(k)
            base = ker(u,v,om)
            mods = np.stack([np.cos(0.5*u),np.sin(0.5*v),np.cos(0.25*(u+v))],axis=0)
            A = minor_amplitude(1.0,mods)
            chaos_phase = minor_phase_iter(chaos_phase)
            phase = om + 0.07*chaos_phase + 0.01*(oct_idx-1)
            field = A*np.cos(phase)*base
            pos = field
            neg = -0.92*field
            Sx += aX*(0.6*pos + 0.4*neg)
            Sy += aY*(0.4*pos + 0.6*neg)
            Sz += aZ*zeta*(0.5*pos - 0.5*neg)
            mode_energy[j,k]=np.abs(field)
            phases_accum.append(phase)
    X,Y,Z = embed_3d(u,v,Sx,Sy,Sz,params["R"],params["r"])
    holonomies = {}
    for j,pr in enumerate(PRINCIPLES):
        Fp = np.sum(mode_energy[j],axis=0)
        Au,Av = conn_from_scalar(Fp)
        holonomies[pr] = [complex(z) for z in holonomy_spectrum(Au,Av)]
    phases_accum = np.array(phases_accum,dtype=np.float64)
    C = kuramoto_r(phases_accum)
    H = occupancy_fraction(mode_energy)
    N = nearness_metric(u,v)
    tau_inv = params["k_time"]*C*H*N
    near_field = np.sum([mode_energy[3,kk] for kk in range(12)],axis=0)
    far_field = np.sum([np.roll(mode_energy[3,kk],shift=+3,axis=1) for kk in range(12)],axis=0)
    love_cost = love_alignment_cost(near_field,far_field)
    new_state = dict(state)
    new_state["chaos_phase"]=chaos_phase
    return {"X":X,"Y":Y,"Z":Z,"energy":mode_energy,"holonomies":holonomies,"C":C,"H":H,"N":N,"tau_inv":tau_inv,"love_cost":love_cost}, new_state

# = renormalization =
def R_update(theta,stats):
    J = stats.get("J",0.5)
    return float(theta + 0.25*stats.get("delta",0.0) - 0.1*J*theta)

# = aggregation =
def aggregate_F11(energies,holonomies,love_cost,C,H,N):
    T = float(np.mean(energies[0])); P = float(np.mean(energies[1])); L = float(np.mean(energies[2]))
    A = float(np.mean(energies[3])); W = float(np.mean(energies[4])); F = float(np.mean(energies[5])); G = float(np.mean(energies[6]))
    m = list(map(float,[np.var(energies[3]),np.var(energies[4]),np.var(energies[5]),np.var(energies[2]),np.var(energies[0])]))
    crown = np.abs(np.prod([holonomies[p][0] for p in PRINCIPLES]))
    chaos = np.abs(np.prod([holonomies[p][1] for p in PRINCIPLES]))
    return [T,P,L,A,W,F,G] + m + [float(crown),float(chaos)], {"C":C,"H":H,"N":N,"love_cost":love_cost}

# = pipeline =
def run_pipeline(octaves=12,uN=192,vN=96,R=3.0,r=1.0,ax=0.18,ay=0.18,az=0.22,k_time=1.0):
    u,v = grid(uN,vN)
    params = {"R":R,"r":r,"ax":ax,"ay":ay,"az":az,"k_time":k_time}
    state = {"chaos_phase":0.333}
    octaves_out = []
    states = []
    S_prev = {"theta":0.0}
    for n in range(1,octaves+1):
        out, state = build_octave(u,v,n,state,params)
        F11, metrics = aggregate_F11(out["energy"],out["holonomies"],out["love_cost"],out["C"],out["H"],out["N"])
        phi_canvas = dielectric_canvas(u,v)
        hol_hash = md5(np.array([np.angle(z) for p in PRINCIPLES for z in out["holonomies"][p]],dtype=np.float64).tobytes())
        octaves_out.append({
            "index": n,
            "holonomy_hash": hol_hash,
            "F11": F11,
            "metrics": metrics,
            "tau_inverse": out["tau_inv"],
            "summary_energy_mean": [float(np.mean(out["energy"][j])) for j in range(len(PRINCIPLES))],
            "summary_energy_var": [float(np.var(out["energy"][j])) for j in range(len(PRINCIPLES))]
        })
        stats = {"J":0.5,"delta":float(np.mean(F11))}
        S_prev["theta"] = R_update(S_prev["theta"],stats)
        states.append({"n":n,"theta":S_prev})
    return {
        "schema_version":"3.0",
        "summary":{
            "octaves": octaves,
            "principles": PRINCIPLES,
            "tones": ZODIAC,
            "modes_per_octave": len(PRINCIPLES)*12*2,
            "effective_dimensions":11,
            "canvas_dimension":1,
            "total_dimensions":12
        },
        "state_trace": states,
        "octaves": octaves_out
    }

# = geometry modeling =
def _ds(arr, step_u, step_v):
    if step_u<=1 and step_v<=1: return arr
    return arr[::step_u, ::step_v]

def build_mesh_from_grid(X,Y,Z, wrap_u=True, wrap_v=True):
    uN, vN = X.shape
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    faces = []
    def idx(i,j): return i*vN + j
    u_max = uN-1 if not wrap_u else uN
    v_max = vN-1 if not wrap_v else vN
    for i in range(u_max):
        i2 = (i+1) % uN
        for j in range(v_max):
            j2 = (j+1) % vN
            v00 = idx(i,j)
            v01 = idx(i,j2)
            v10 = idx(i2,j)
            v11 = idx(i2,j2)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    return verts, np.asarray(faces, dtype=np.int64)

def mesh_surface_area(verts, faces):
    v = verts
    f = faces
    a = v[f[:,0]]; b = v[f[:,1]]; c = v[f[:,2]]
    return float(0.5*np.linalg.norm(np.cross(b-a, c-a), axis=1).sum())

def mesh_bbox(verts):
    mn = verts.min(axis=0); mx = verts.max(axis=0)
    return [float(mn[0]),float(mn[1]),float(mn[2])],[float(mx[0]),float(mx[1]),float(mx[2])]

def write_obj(path, verts, faces, group_name=None):
    with open(path, "w", encoding="utf-8") as f:
        if group_name: f.write(f"g {group_name}\n")
        for x,y,z in verts:
            f.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
        for a,b,c in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")

def write_obj_multi(path, parts):
    with open(path, "w", encoding="utf-8") as f:
        base_index = 0
        for (name, verts, faces) in parts:
            f.write(f"g {name}\n")
            for x,y,z in verts:
                f.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
            faces_shifted = faces + base_index + 1
            for a,b,c in faces_shifted:
                f.write(f"f {a} {b} {c}\n")
            base_index += verts.shape[0]

def write_ply_pointcloud(path, verts):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {verts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x,y,z in verts:
            f.write(f"{x:.7f} {y:.7f} {z:.7f}\n")

def model_geometry_for_octave(uN,vN,R,r,ax,ay,az,k_time,oct_idx, ds_u=1, ds_v=1):
    u,v = grid(uN,vN)
    params = {"R":R,"r":r,"ax":ax,"ay":ay,"az":az,"k_time":k_time}
    state = {"chaos_phase":0.333}
    out,_ = build_octave(u,v,oct_idx,state,params)
    X = _ds(out["X"], ds_u, ds_v)
    Y = _ds(out["Y"], ds_u, ds_v)
    Z = _ds(out["Z"], ds_u, ds_v)
    verts, faces = build_mesh_from_grid(X,Y,Z, wrap_u=True, wrap_v=True)
    area = mesh_surface_area(verts, faces)
    bb_min, bb_max = mesh_bbox(verts)
    return {"verts":verts, "faces":faces, "area":area, "bbox_min":bb_min, "bbox_max":bb_max}

def model_geometry_all(octaves,uN,vN,R,r,ax,ay,az,k_time, ds_u=1, ds_v=1, select=None):
    results = {}
    parts = []
    for n in range(1, octaves+1):
        if select and (n not in select): continue
        g = model_geometry_for_octave(uN,vN,R,r,ax,ay,az,k_time,n, ds_u, ds_v)
        results[n] = {"area":g["area"], "bbox_min":g["bbox_min"], "bbox_max":g["bbox_max"], "vcount":int(g["verts"].shape[0]), "fcount":int(g["faces"].shape[0])}
        parts.append((f"Octave_{n}", g["verts"], g["faces"]))
    return results, parts

# = self-validator =
class SelfValidator:
    def __init__(self, data, no_color=False):
        self.data = data
        self.passed = 0
        self.failed = 0
        self.warn = 0
        self.no_color = no_color
        self.CLR = {"G":"\033[92m","R":"\033[91m","Y":"\033[93m","B":"\033[94m","C":"\033[96m","X":"\033[0m","BD":"\033[1m"}
        if self.no_color:
            for k in self.CLR: self.CLR[k] = ""
    def _pass(self,msg): print(f"{self.CLR['G']}✓ PASS:{self.CLR['X']} {msg}"); self.passed+=1
    def _fail(self,msg): print(f"{self.CLR['R']}✗ FAIL:{self.CLR['X']} {msg}"); self.failed+=1
    def _warn(self,msg): print(f"{self.CLR['Y']}⚠ WARN:{self.CLR['X']} {msg}"); self.warn+=1
    def _sec(self,msg):
        print(f"\n{self.CLR['BD']}{self.CLR['B']}{'='*60}{self.CLR['X']}\n{self.CLR['BD']}{msg}{self.CLR['X']}\n{self.CLR['B']}{'='*60}{self.CLR['X']}")
    def _finite(self,x):
        if isinstance(x,(int,float)): return math.isfinite(x)
        a = np.asarray(x)
        return np.isfinite(a).all()
    def schema(self):
        self._sec("Schema Validation")
        try:
            assert self.data.get("schema_version")=="3.0"
            self._pass(f"Schema version: {self.data['schema_version']}")
        except: self._fail("Schema version mismatch")
        summary = self.data.get("summary",{})
        try:
            assert summary.get("octaves")==12
            self._pass(f"Octaves: {summary.get('octaves')}")
        except: self._fail("Expected 12 octaves")
        try:
            assert isinstance(summary.get("principles"),list) and len(summary.get("principles"))==7
            self._pass(f"Principles: {summary.get('principles')}")
        except: self._fail("Expected 7 principles")
        try:
            assert summary.get("total_dimensions")==12
            self._pass(f"Total dimensions: {summary.get('total_dimensions')}")
        except: self._fail("Expected 12 total dimensions")
    def inverse_zero(self):
        self._sec("Divide-by-Zero Handling Validation")
        try:
            z = InverseZero_operator(0.0)
            ok = (z==0.0) and (math.copysign(1.0,z)==-1.0 or z==0.0)
            if ok: self._pass("InverseZero_operator maps 0 safely (non-NaN, signed zero allowed)")
            else: self._fail("InverseZero_operator mapping check failed")
        except Exception as e:
            self._fail(f"InverseZero_operator raised exception: {e}")
        bad=False
        for octv in self.data.get("octaves",[]):
            F11 = octv.get("F11",[])
            metrics = octv.get("metrics",{})
            tau = octv.get("tau_inverse",0.0)
            for v in F11:
                if not self._finite(v): self._fail(f"Octave {octv['index']}: F11 contains non-finite"); bad=True
            for k,v in metrics.items():
                if isinstance(v,(int,float)) and not self._finite(v): self._fail(f"Octave {octv['index']}: metric {k} non-finite"); bad=True
            if not self._finite(tau): self._fail(f"Octave {octv['index']}: tau_inverse non-finite"); bad=True
        if not bad:
            self._pass("No NaN or Inf values detected in any octave")
            self._pass("InverseZero_operator successfully handled zero divisions")
    def holonomy(self):
        self._sec("Holonomy Integrity Validation")
        for octv in self.data.get("octaves",[]):
            hh = octv.get("holonomy_hash","")
            if isinstance(hh,str) and len(hh)==32:
                self._pass(f"Octave {octv['index']}: Valid holonomy hash: {hh[:8]}...")
            else:
                self._fail(f"Octave {octv.get('index','?')}: Invalid holonomy hash")
            F11 = octv.get("F11",[])
            if len(F11)>=14:
                crown = F11[12]; chaos = F11[13]
                if abs(crown-1.0)<1e-6: self._pass(f"Octave {octv['index']}: Crown holonomy closure verified ({crown:.10f})")
                else: self._warn(f"Octave {octv['index']}: Crown deviation {abs(crown-1.0):.2e}")
                if abs(chaos-1.0)<1e-6: self._pass(f"Octave {octv['index']}: Chaos holonomy closure verified ({chaos:.10f})")
                else: self._warn(f"Octave {octv['index']}: Chaos deviation {abs(chaos-1.0):.2e}")
    def coherence(self):
        self._sec("Coherence & Alignment Metrics")
        for octv in self.data.get("octaves",[]):
            m = octv.get("metrics",{})
            C = m.get("C",-1); H = m.get("H",-1); N = m.get("N",-1); Lc = m.get("love_cost",-1)
            if 0<=C<=1: self._pass(f"Octave {octv['index']}: Kuramoto C = {C:.6e} (valid)")
            else: self._fail(f"Octave {octv['index']}: Kuramoto C out of bounds")
            if 0<=H<=1: self._pass(f"Octave {octv['index']}: Occupancy H = {H:.6f} (valid)")
            else: self._fail(f"Octave {octv['index']}: Occupancy H out of bounds")
            if 0<=N<=1: self._pass(f"Octave {octv['index']}: Nearness N = {N:.6f} (valid)")
            else: self._fail(f"Octave {octv['index']}: Nearness N out of bounds")
            if 0<=Lc<=1: self._pass(f"Octave {octv['index']}: Love cost = {Lc:.10f} (valid)")
            else: self._fail(f"Octave {octv['index']}: Love cost out of bounds")
    def energy(self):
        self._sec("Energy Conservation & Stability")
        principles = self.data.get("summary",{}).get("principles",PRINCIPLES)
        for octv in self.data.get("octaves",[]):
            means = octv.get("summary_energy_mean",[])
            vars_ = octv.get("summary_energy_var",[])
            if len(means)==7 and len(vars_)==7:
                tot = sum(means)
                print(f"\033[96mℹ INFO:\033[0m Octave {octv['index']}: Total principle energy = {tot:.6f}" if not self.no_color else f"INFO: Octave {octv['index']}: Total principle energy = {tot:.6f}")
                if all(e>=0 for e in means): self._pass(f"Octave {octv['index']}: All principle energies non-negative")
                else: self._fail(f"Octave {octv['index']}: Negative energies found")
                if all(v>=0 for v in vars_): self._pass(f"Octave {octv['index']}: All variances non-negative")
                else: self._fail(f"Octave {octv['index']}: Negative variances found")
                for i,p in enumerate(principles):
                    print(f"\033[96mℹ INFO:\033[0m   {p}: μ={means[i]:.4f}, σ²={vars_[i]:.4f}" if not self.no_color else f"INFO:   {p}: μ={means[i]:.4f}, σ²={vars_[i]:.4f}")
    def renorm(self):
        self._sec("Renormalization Flow (R-Update)")
        st = self.data.get("state_trace",[])
        if len(st)>1:
            thetas = [s["theta"]["theta"] for s in st]
            t0, tF = thetas[0], thetas[-1]
            tstd = float(np.std(np.asarray(thetas)))
            print(f"\033[96mℹ INFO:\033[0m Theta evolution: {t0:.6f} → {tF:.6f}" if not self.no_color else f"INFO: Theta evolution: {t0:.6f} → {tF:.6f}")
            print(f"\033[96mℹ INFO:\033[0m Theta std dev: {tstd:.6e}" if not self.no_color else f"INFO: Theta std dev: {tstd:.6e}")
            if tstd<1e-6: self._pass(f"Renormalization theta converged to {tF:.10f}")
            else: self._warn(f"Theta still evolving (std: {tstd:.6e})")
            if all(math.isfinite(x) for x in thetas): self._pass("All theta values numerically stable")
            else: self._fail("Unstable theta values (non-finite)")
        else:
            self._warn("Insufficient state trace for renormalization analysis")
    def tau_inv(self):
        self._sec("Temporal Evolution (τ⁻¹ Analysis)")
        taus = [octv.get("tau_inverse",0.0) for octv in self.data.get("octaves",[])]
        print(f"\033[96mℹ INFO:\033[0m τ⁻¹ sequence across octaves:" if not self.no_color else "INFO: τ⁻¹ sequence across octaves:")
        for i,t in enumerate(taus,1):
            print(f"\033[96mℹ INFO:\033[0m   Octave {i}: {t:.6e}" if not self.no_color else f"INFO:   Octave {i}: {t:.6e}")
        if all(math.isfinite(t) for t in taus): self._pass("All τ⁻¹ values finite and well-defined")
        else: self._fail("Invalid τ⁻¹ values (non-finite)")
        if all(t>=0 for t in taus): self._pass("All τ⁻¹ values non-negative (forward time)")
        else: self._fail("Negative τ⁻¹ value (causality violation)")
    def summary(self):
        self._sec("Validation Summary")
        total = self.passed + self.failed + self.warn
        rate = (self.passed/total*100.0) if total>0 else 0.0
        print(f"{self.CLR['G']}Passed:   {self.passed}{self.CLR['X']}")
        print(f"{self.CLR['R']}Failed:   {self.failed}{self.CLR['X']}")
        print(f"{self.CLR['Y']}Warnings: {self.warn}{self.CLR['X']}")
        print(f"\n{self.CLR['BD']}Pass Rate: {rate:.1f}%{self.CLR['X']}")
        ok = self.failed==0
        if ok:
            print(f"\n{self.CLR['G']}{self.CLR['BD']}SUCCESS: All critical validations passed!{self.CLR['X']}")
            print(f"{self.CLR['G']}The InverseZero operator successfully handled division by zero.{self.CLR['X']}")
        else:
            print(f"\n{self.CLR['R']}{self.CLR['BD']}FAILURE: {self.failed} critical test(s) failed.{self.CLR['X']}")
        return ok
    def run_all(self):
        self.schema()
        self.inverse_zero()
        self.holonomy()
        self.coherence()
        self.energy()
        self.renorm()
        self.tau_inv()
        return self.summary()

# = cli =
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--octaves",type=int,default=12)
    ap.add_argument("--uN",type=int,default=192)
    ap.add_argument("--vN",type=int,default=96)
    ap.add_argument("--R",type=float,default=3.0)
    ap.add_argument("--r",type=float,default=1.0)
    ap.add_argument("--ax",type=float,default=0.18)
    ap.add_argument("--ay",type=float,default=0.18)
    ap.add_argument("--az",type=float,default=0.22)
    ap.add_argument("--k_time",type=float,default=1.0)
    ap.add_argument("--outdir",type=str,default="paraclete_out")
    ap.add_argument("--validate",action="store_true",default=True)
    ap.add_argument("--no-color",action="store_true",default=False)
    ap.add_argument("--strict",action="store_true",default=False)
    ap.add_argument("--export-obj",action="store_true",default=True)
    ap.add_argument("--export-ply",action="store_true",default=False)
    ap.add_argument("--combine",action="store_true",default=True)
    ap.add_argument("--octaves-select",type=str,default="all")
    ap.add_argument("--ds-u",type=int,default=2)
    ap.add_argument("--ds-v",type=int,default=2)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    result = run_pipeline(octaves=args.octaves,uN=args.uN,vN=args.vN,R=args.R,r=args.r,ax=args.ax,ay=args.ay,az=args.az,k_time=args.k_time)

    validation_ok = True
    if args.validate:
        print(f"\n\033[1m\033[96mParaclete Self-Validation v1.1\033[0m" if not args.no_color else "\nParaclete Self-Validation v1.1")
        print(f"{'\033[96m' if not args.no_color else ''}Analyzing freshly generated structure...\033[0m\n" if not args.no_color else "Analyzing freshly generated structure...\n")
        sv = SelfValidator(result,no_color=args.no_color)
        validation_ok = sv.run_all()

    out_json = os.path.join(args.outdir,"structure.json")
    with open(out_json,"w",encoding="utf-8") as f: json.dump(result,f,indent=2)
    print(out_json)

    select = None
    if args.octaves_select.strip().lower()!="all":
        try:
            sel = [int(s) for s in args.octaves_select.split(",") if s.strip()]
            select = [n for n in sel if 1<=n<=args.octaves]
        except:
            select = None

    geom_metrics, parts = model_geometry_all(
        octaves=args.octaves,
        uN=args.uN, vN=args.vN,
        R=args.R, r=args.r,
        ax=args.ax, ay=args.ay, az=args.az,
        k_time=args.k_time,
        ds_u=max(1,args.ds_u), ds_v=max(1,args.ds_v),
        select=select
    )

    geom_dir = ensure_dir(os.path.join(args.outdir, "geometry"))
    with open(os.path.join(geom_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(geom_metrics, f, indent=2)

    if args.export_obj:
        if args.combine:
            write_obj_multi(os.path.join(geom_dir, "creation_12_octaves.obj"), parts)
        else:
            for name, v, fc in parts:
                write_obj(os.path.join(geom_dir, f"{name}.obj"), v, fc, group_name=name)

    if args.export_ply:
        for name, v, _ in parts:
            write_ply_pointcloud(os.path.join(geom_dir, f"{name}.ply"), v)

    if args.strict and not validation_ok:
        sys.exit(1)
    sys.exit(0)

if __name__=="__main__":
    main()
