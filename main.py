from collections import deque
import numpy as np
import cv2
import numba
import multiprocessing as mp
try:
    from multiprocessing.connection import Connection as conn
except ImportError:
    from multiprocessing.connection import PipeConnection as conn
from multiprocessing import shared_memory as mpsm
from time import sleep
from random import shuffle

G = np.float64(6.6743e-11)
FORCE_TIME_SCALE = 1
dt = FORCE_TIME_SCALE
one_over_dt_squared = 1/(dt**2)
SCREEN_RATIO = 1920/1080
CANVAS_H = 1000
CANVAS_W = int(CANVAS_H*SCREEN_RATIO)
MAX_PARTICLES = 50
FRAMES_PER_SECOND = 100
FRAME_DELAY = 1000/FRAMES_PER_SECOND # cv2.waitKey measures in milliseconds, so 1000 is one second
DESIRED_ANIMATION_LENGTH_SECONDS = 10
FRAMES_TO_COMPUTE = FRAMES_PER_SECOND*DESIRED_ANIMATION_LENGTH_SECONDS

start = 10
stopw = CANVAS_W-start
stepw = (stopw-start)//100
stoph = CANVAS_H-start
steph = (stoph-start)//100


@numba.njit(parallel=True,nogil=True,fastmath=True)
def draw_particles(canvas:np.ndarray,mass:np.ndarray,position:np.ndarray):
    # update canvas so that past point positions appear as a fading color trail
    for y in numba.prange(canvas.shape[0]):
        for x in canvas[y]:
            abc = 255-x
            if abc.sum()>0:
                x[0] += min(20,abc[0])
                x[1] += min(10,abc[1])
                x[2] += min(5,abc[2])
    # now place the new particles
    for i in numba.prange(MAX_PARTICLES):
        y,x = position[i]
        _m = np.ceil(mass[i,0]/2)
        m = np.int64(_m)
        center = np.array((y,x))
        for _y in range(y-m,y+m):
            for _x in range(x-m,x+m):
                pt = np.array((_y,_x))
                if 0<=_y<CANVAS_H and 0<=_x<CANVAS_W:
                    if np.sqrt(np.square(pt-center).sum())<=m:
                        canvas[_y,_x] = 0,0,0


def compute_frames(pipe_end:conn,mem_name:str):
    @numba.njit(nogil=True,fastmath=True)
    def particle_update_helper(positions:np.ndarray,
                               mass:np.ndarray,
                               vel:np.ndarray,
                               acc:np.ndarray,
                               random_insertion_positions:np.ndarray):
        mass_argsorted = np.argsort(mass[:, 0])[::-1]
        mass = mass[mass_argsorted]
        positions = positions[mass_argsorted]
        vel = vel[mass_argsorted]
        acc = acc[mass_argsorted]
        collision_rad_in = np.ceil(mass / 2).astype(np.uint16)
        xy_zero = np.zeros((2,), dtype=np.float64)
        # first we combine particles that have colided, and remove those that
        # have left the canvas. Replacing all combined/removed particles with
        # new ones at the edge of the canvas.
        consumed_ptcls = set()
        for i in range(MAX_PARTICLES - 1):
            if i not in consumed_ptcls:
                if 0 <= positions[i, 0] < CANVAS_H and 0 <= positions[i, 1] < CANVAS_W:
                    elim_dists = np.sqrt(np.square(positions[i + 1:] - positions[i]).sum(-1))
                    for j in range(i + 1, MAX_PARTICLES - 1):
                        elim_dist = elim_dists[j - i]
                        collision_rad = collision_rad_in[j, 0] + collision_rad_in[i, 0]
                        if elim_dist <= collision_rad:
                            # we have particle collision, and need to combine mass, acceleration, and velocity
                            mass_ratio = mass[j, 0] / mass[i, 0]
                            mass[i] += mass[j]
                            vel[i] += vel[j] * mass_ratio
                            acc[i] += acc[j] * mass_ratio
                            # now we record the position of the consumed particle and replace it later with a new random
                            # particle placed somewhere around the edge of the canvas.
                            consumed_ptcls.add(j)
                else:
                    # The particle has left the canvas and needs to have its acceleration and velocity set to 0
                    # so that it can be attracted back onto the canvas
                    vel[i] *= xy_zero
                    acc[i] *= xy_zero
        if not consumed_ptcls:
            consumed_ptcls.add(0)
        if consumed_ptcls:
            consumed_ptcls_count = len(consumed_ptcls)
            rand_masses = np.random.random_sample(consumed_ptcls_count) * 10.
            for i, consumed in enumerate(consumed_ptcls):
                mass[consumed] = rand_masses[i]
                positions[consumed] = random_insertion_positions[i % random_insertion_positions.shape[0]]
                vel[consumed,:] *= 0
                acc[consumed,:] *= 0
    @numba.njit(parallel=True,nogil=True,fastmath=True)
    def do_particle_update(positions:np.ndarray,
                           mass:np.ndarray,
                           vel:np.ndarray,
                           acc:np.ndarray,
                           random_insertion_positions:np.ndarray):
        particle_update_helper(positions,mass,vel,acc,random_insertion_positions)
        out_pos = np.empty_like(positions)
        out_vel = np.empty_like(vel)
        out_acc = np.empty_like(acc)
        out_mass = mass.copy()
        for i in numba.prange(MAX_PARTICLES):
            mask = np.ones((MAX_PARTICLES,),dtype=np.bool_)
            mask[i] = False
            ip = positions[i]
            im = mass[i,0]
            ia = acc[i]
            iv = vel[i]
            ptkl_dist = (positions[mask] - ip).astype(np.float64)
            r_squared = np.square(ptkl_dist).sum(-1)
            r = np.sqrt(r_squared)
            a_ratios = (ptkl_dist.astype(np.float32).T/r)
            # Mp = mass[mask,0]*im
            iG = G*r
            iG *= one_over_dt_squared
            a = iG
            Ma = mass[mask,0]+im
            a *= Ma
            a_sum = (a*a_ratios).T.sum(0)
            out_acc[i] = ia + a_sum
            out_vel[i] = iv+out_acc[i]
            out_pos[i] = ip+out_vel[i]
            mask[i] = True
        return out_pos,out_mass,out_vel,out_acc

    tmp = []
    for y in (start, stoph):
        for x in range(start, stopw, stepw):
            tmp.append((y, x))
    for x in (start, stopw):
        for y in range(start + steph, stoph, steph):
            tmp.append((y, x))
    shuffle(tmp)
    insertion_positions = np.array(tmp,dtype=np.int32)
    del tmp
    shared_mem = mpsm.SharedMemory(name=mem_name, create=False)
    try:
        canvas = np.ndarray((CANVAS_H,CANVAS_W,3),dtype=np.uint8,buffer=shared_mem.buf)
        canvas[:,:,:] *= 0
        canvas[:,:,:] += 255
        input_mass = np.zeros((MAX_PARTICLES, 1), dtype=np.float64) + np.random.random((MAX_PARTICLES, 1)) * 10
        input_position = np.zeros((MAX_PARTICLES, 2), dtype=np.int64) \
                   + np.int32(np.round(np.random.random((MAX_PARTICLES, 2)) * (CANVAS_H - 101) + 100))
        input_velocity = np.zeros((MAX_PARTICLES, 2), dtype=np.float64)
        input_accel = np.zeros_like(input_velocity)
        draw_particles(canvas, input_mass, input_position)
        pipe_end.send(0)
        signal = pipe_end.recv()
        while signal>0:
            input_position,input_mass,input_velocity,input_accel = \
                do_particle_update(input_position,input_mass,input_velocity,input_accel,insertion_positions)
            draw_particles(canvas, input_mass, input_position)
            if pipe_end.closed:
                break
            pipe_end.send(signal)
            signal = pipe_end.recv()
    finally:
        shared_mem.close()



def main():
    winname = "canvas"
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname,CANVAS_W,CANVAS_H)
    frame_deq = deque(maxlen=10)
    frames_computed = 1
    mem_name = "canvas"
    parent, child = mp.Pipe()
    proc = mp.Process(target=compute_frames,args=(child,mem_name))
    shared_mem = mpsm.SharedMemory(name=mem_name, create=True,
                                   size=CANVAS_H*CANVAS_W*3)
    try:
        proc.start()
        parent.poll(None)
        parent.recv()
        canvas = np.ndarray((CANVAS_H,CANVAS_W,3),dtype=np.uint8,buffer=shared_mem.buf)
        while frames_computed < FRAMES_TO_COMPUTE:
            cv2.imshow(winname,canvas)
            delay = int(max(FRAME_DELAY-np.ceil(FRAME_DELAY*len(frame_deq)/10),1))
            cv2.setWindowTitle(winname,f"{winname}: {frames_computed:0>5}, {delay}")
            cv2.waitKey(delay)
            parent.send(frames_computed)
            parent.poll(None)
            parent.recv()
            frames_computed += 1
        parent.send(0)
    finally:
        shared_mem.close()
        shared_mem.unlink()
        parent.close()
        child.close()
        proc.join()
        proc.close()

if __name__ == '__main__':
    main()

