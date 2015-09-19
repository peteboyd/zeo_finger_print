#!/usr/bin/env python
import os
from os.path import join, dirname, realpath
import sys
import numpy as np
from time import time
import platform
import tarfile
sys.path[:0] = [os.path.expandvars('$HOME/modules/faps'),
                join(dirname(realpath(__file__)),
                     "build","lib.%s-%s-%i.%i"%(platform.system().lower(), 
                                                platform.machine(), 
                                                sys.version_info.major, 
                                                sys.version_info.minor))]
from SphereCollision import compute_collision_array 
from faps import Structure, Atom, Cell
from elements import UFF
from scipy.spatial import Voronoi, distance
from scipy import linalg
import matplotlib.pyplot as plt
import math
import itertools
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch

cutoff=10
probe=1.86
def get_exclude_dists(voro,rads, xyzarray):
    badids=[]

    for j in range(xyzarray.shape[0]):
        for k in range(j+1, xyzarray.shape[0]):
            p1 = xyzarray[j]
            p2 = xyzarray[k]
            p2p1 = p2-p1

            for vid, v in enumerate(voro):
                u=np.dot(v-p1, p2p1)/np.dot(p2p1,p2p1)
                if u>=0. and u<=1.:
                    p = u*p2p1
                    if np.linalg.norm(p-v) <= rads[vid]:
                        badids.append((j,k))
    return badids

def minimum_supercell(cell, cutoff):
    a_cross_b = np.cross(cell[0], cell[1])
    b_cross_c = np.cross(cell[1], cell[2])
    c_cross_a = np.cross(cell[2], cell[0])

    vol = np.dot(cell[0], b_cross_c)

    widths = [vol/np.linalg.norm(b_cross_c),
              vol/np.linalg.norm(c_cross_a),
              vol/np.linalg.norm(a_cross_b)]
    return tuple(int(math.ceil(2*cutoff/x)) for x in widths)


def main():
    workdir = os.getcwd()
    targzfiles = [i for i in os.listdir(workdir) if os.path.isfile(os.path.join(workdir, i)) and i.endswith(".tar.gz")]
    try: 
        if sys.argv[1].endswith('.tar.gz'):
            targzfiles=[sys.argv[1]]
    except IndexError:
        pass

    times = time()
    atom_distribution = {}
    atom_type_distribution = {}
    ltfive = []
    gtfiveltthousand = []
    gtthousand = []
    atomcount_distrib = []

    # this is just a test
    tarball = tarfile.open(os.path.join(workdir, targzfiles[0]), 'r:gz')
    mofname = tarball.next()
    print(mofname.name[:-4])
    cif = tarball.extractfile(mofname).read().decode('utf=8')
    mof = Structure(name=mofname.name[:-4])
    mof.from_cif(string=cif)
    cell = mof.cell.cell
    icell = mof.cell.inverse
    min_supercell = minimum_supercell(mof.cell.cell, cutoff)
    #min_supercell=(1,1,1)
    supercell = np.multiply(mof.cell.cell.T, min_supercell).T
    isupercell = np.linalg.inv(supercell).T

    trans = np.mean(supercell, axis=0) - np.mean(mof.cell.cell, axis=0)
    natms = len(mof.atoms)
    xyzcoords = np.empty((natms*np.product(min_supercell),3))
    elements = []
    cells = list(itertools.product(*[itertools.product(range(j)) for j in min_supercell]))
    dcount = 0
    for id, box in enumerate(cells):
        box = tuple([j[0] for j in box])
        v = np.dot(cell.T, box).T
        for idx, atom in enumerate(mof.atoms):
            xyzcoords[dcount] = np.dot(np.dot(isupercell, atom.pos+trans+v) % 1., supercell) 
            #xyzcoords[dcount] = atom.pos.copy()
            elements.append(atom.type)
            dcount += 1

    vor=Voronoi(xyzcoords)
    verts = vor.vertices.copy()
    rem = []
    for id, j in enumerate(verts):
        i = np.dot(isupercell,j)
        if np.where(i > 1.)[0].shape[0] or np.where(i < 0.)[0].shape[0]:
            rem.append(id)

    verts = np.delete(verts, rem, axis=0)
    rem = []
    v_a_dists = distance.cdist(verts, xyzcoords)
    v_rads = v_a_dists.min(axis=1)
    vids = np.argmin(v_a_dists, axis=1)
    uff_rads = np.array([UFF[i][0] for i in elements])

    v_rads -= uff_rads[vids]
    rem = np.where((v_rads-probe) <= 0.)[0]
    #for (vid,aid), dist in np.ndenumerate(v_a_dists):
    #    uff_rad = UFF[elements[aid]][0]
    #    if dist <= (uff_rad + probe):
    #        rem.append(vid)
    verts = np.delete(verts, rem, axis=0)
    v_rads = np.delete(v_rads, rem, axis=0)
    #excl_dists = get_exclude_dists(verts, v_rads, xyzcoords)
    collisions = compute_collision_array(xyzcoords, verts, v_rads)
    pair_dist = distance.cdist(xyzcoords, xyzcoords)
    colinds = np.where(collisions>0)

    timef=time()
    print("Walltime: %.2f seconds"%(timef-times))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sf=10.
    count =0 
    for v1,v2 in zip(*colinds):
        if count == 100:
            break
        p1, p2 = xyzcoords[v1], xyzcoords[v2]
        length = np.linalg.norm(p2-p1)
        dv = (p2-p1)/length
        ax.quiver(p2[0], p2[1], p2[2], dv[0], dv[1], dv[2], length=length, arrow_length_ratio=0.09, color='k')
        count += 1
    ax.scatter(xyzcoords[:,0], xyzcoords[:,1], xyzcoords[:,2], c='b', s=uff_rads*sf)
    ax.scatter(verts[:,0], verts[:,1], verts[:,2], c='r', s=v_rads*sf)

    plt.show()
if __name__=="__main__":
    main()
