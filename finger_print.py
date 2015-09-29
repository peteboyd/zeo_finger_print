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
from SphereCollision import minimg_distances, compute_collision_array 
from faps import Structure, Atom, Cell
from elements import UFF, ATOMIC_NUMBER
from scipy.spatial import Voronoi, distance
from scipy import linalg
import matplotlib.pyplot as plt
import math
import pickle
import itertools
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch

cutoff = 15.
nbins = 30
probe=1.86
class Hologram(object):

    def __init__(self, name="Default"):
        self.name = name
        self.atomic_bins = {1:0,
                            6:1,
                            7:2,
                            8:3,
                            9:4,
                            16:5,
                            17:6,
                            30:7,
                            35:8,
                            53:9}
        self.distance_bins, self.deltabin = np.linspace(0, cutoff, nbins+1, retstep=True)
        
        self.similarity_func = None

        self.hologram=np.zeros((len(self.atomic_bins.keys()),
                                 len(self.atomic_bins.keys()),
                                 nbins+1))

    def construct_hologram(self, pair_dist, collisions, elements, natms):
        for j in range(natms):
            for k in range(j+1, natms):
                dist = pair_dist[j,k]
                #print("distance: %.2f"%dist)
                if (j!=k) and (dist <= cutoff) and (not collisions[j,k]):
                    nums = sorted([ATOMIC_NUMBER.index(elements[j]), ATOMIC_NUMBER.index(elements[k])])
                    nums.reverse()
                    distbin1 = math.floor(dist/cutoff*nbins)
                    distbin2 = math.ceil(dist/cutoff*nbins)
                    distfrac1 = 1.-np.abs(self.distance_bins[distbin1] - dist)/self.deltabin
                    distfrac2 = 1.-np.abs(self.distance_bins[distbin2] - dist)/self.deltabin
                    #print("dist bin: %i; dist frac: %.2f"%(distbin1, distfrac1))
                    #print("dist bin: %i; dist frac: %.2f"%(distbin2, distfrac2))
                    self.hologram[self.atomic_bins[nums[0]],
                              self.atomic_bins[nums[1]], 
                              distbin1] += distfrac1
                    self.hologram[self.atomic_bins[nums[0]],
                              self.atomic_bins[nums[1]], 
                              distbin2] += distfrac2

    def set_similarity_metric(self,val="mtw_cont"):
        """Following metrics for similarity testing:
        mtw_cont (default)
        mtw_bin
        mtu_cont
        mtu_bin
        tan_bin
        tan_cont
        tan_abs_bin
        """
        if val == "mtw_cont":
            self.similarity_func = self.mtw_cont 
        elif val == "mtw_bin":
            self.similarity_func = self.mtw_bin
        elif val == "mtu_cont":
            self.similarity_func = self.mtu_cont
        elif val == "mtu_bin":
            self.similarity_func = self.mtu_bin
        elif val == "tan_bin":
            self.similarity_func = self.tan_bin
        elif val == "tan_cont":
            self.similarity_func = self.tan_cont
        elif val == "tan_abs_bin":
            self.similarity_func = self.tan_abs_bin

    def __or__(self, other):
        """Tanimoto similarity"""
        return self.similarity_func(other)

    def mtw_bin(self, other):
        p = self.p(other)
        return 1./3.*(self.tan_bin(other)*(2.-p) + self.tan_abs_bin(other)*(1.+p))
    
    def mtu_bin(self, other):
        return 0.5*(self.tan_bin(other) + self.tan_abs_bin(other))

    def mtw_cont(self, other):
        p = self.p(other)
        return 1./3.*(self.tan_cont(other)*(2.-p) + self.tan_abs_bin(other)*(1.+p))

    def mtu_cont(self, other):
        return 0.5*(self.tan_cont(other) + self.tan_abs_bin(other))

    def tan_bin(self, other):
        a = float(np.array(np.nonzero(self.hologram)).shape[1])
        b = float(np.array(np.nonzero(other.hologram)).shape[1])
        c = float(np.array(
                    np.nonzero(
                        np.logical_and(self.hologram > 0., other.hologram > 0.))).shape[1])
        return  c / (a + b - c)

    def tan_cont(self, other):
        ab = np.sum(np.multiply(self.hologram,other.hologram))
        a = np.sum(np.multiply(self.hologram, self.hologram))
        b = np.sum(np.multiply(other.hologram, other.hologram))
        return ab / (a + b - ab)
    
    def tan_abs_bin(self, other):
        """Just deal with the last dimension"""
        #holo = self.hologram[-1::].flatten()
        #oo = other.hologram[-1::].flatten()
        holo = self.hologram
        oo = other.hologram
        a = float(np.array(np.nonzero(holo)).shape[1])
        b = float(np.array(np.nonzero(oo)).shape[1])
        n = float(holo.size)
        c = float(np.array(np.nonzero(np.logical_and(holo>0., oo>0.))).shape[1]) 
        return (n+c-a-b)/(n-c)

    def p(self, other):
        a = float(np.array(np.nonzero(self.hologram)).shape[1])
        b = float(np.array(np.nonzero(other.hologram)).shape[1])
        n = float(self.hologram.size)
        return (a+b)/(2.*n)

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

def obtain_hologram(mof):
    cell = mof.cell.cell
    icell = mof.cell.inverse
    min_supercell = minimum_supercell(mof.cell.cell, cutoff)
    #min_supercell=(2,2,2)
    # ensure at least 1 periodic image for each dimension 
    #min_supercell = tuple([max(i,j) for i, j in zip(min_supercell, (2,2,2))])
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
    # c script to compute minimum image distances
    #v_a_dists = distance.cdist(verts, xyzcoords)
    v_a_dists = minimg_distances(verts, xyzcoords, supercell)
    v_rads = v_a_dists.min(axis=1)
    vids = np.argmin(v_a_dists, axis=1)
    uff_rads = np.array([UFF[i][0] for i in elements])

    v_rads -= uff_rads[vids]
    rem = np.where((v_rads-probe) <= 0.)[0]
    verts = np.delete(verts, rem, axis=0)
    v_rads = np.delete(v_rads, rem, axis=0)
    #excl_dists = get_exclude_dists(verts, v_rads, xyzcoords)
    collisions = compute_collision_array(xyzcoords, verts, v_rads, supercell)
    #pair_dist = distance.cdist(xyzcoords, xyzcoords)
    pair_dist = minimg_distances(xyzcoords, xyzcoords, supercell)
    holo = Hologram(mof.name)
    holo.construct_hologram(pair_dist, collisions, elements, natms)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #sf=10.
    #count =0 
    #colinds = np.where(collisions>0)
    #for v1,v2 in zip(*colinds):
    #    if count == 100:
    #        break
    #    p1, p2 = xyzcoords[v1], xyzcoords[v2]
    #    length = np.linalg.norm(p2-p1)
    #    dv = (p2-p1)/length
    #    ax.quiver(p2[0], p2[1], p2[2], dv[0], dv[1], dv[2], length=length, arrow_length_ratio=0.09, color='k')
    #    count += 1
    #ax.scatter(xyzcoords[:,0], xyzcoords[:,1], xyzcoords[:,2], c='b', s=uff_rads*sf)
    #ax.scatter(verts[:,0], verts[:,1], verts[:,2], c='r', s=v_rads*sf)

    #plt.show()
    return holo

def mofgen(workdir,files):
    tgzs,cifs = [],[]
    for j in files:
        if j.endswith('tar.gz'):
            tgzs.append(j)

        elif j.endswith('.cif'):
            cifs.append(j)
    for j in tgzs:
        tarball = tarfile.open(os.path.join(workdir, j), 'r:gz')
        mf = tarball.next()
        while mf is not None:
            mofname = mf.name
            mf = tarball.next()
            if mofname.endswith('.cif'):
                cif = tarball.extractfile(mofname).read().decode('utf=8')
                mof = Structure(name=mofname[:-4])
                mof.from_cif(string=cif)
                yield mof
            
    for k in cifs:
        mof = Structure(name=k[:-4])
        mof.from_cif(k)
        yield mof

def main():
    workdir = os.getcwd()
    files = [i for i in os.listdir(workdir) if os.path.isfile(os.path.join(workdir, i)) and i.endswith(".tar.gz") or i.endswith(".cif")]
    try: 
        if sys.argv[1].endswith('.tar.gz') or sys.argv[1].endswith('.cif'):
            files=[sys.argv[1]]
    except IndexError:
        pass

    times = time()
    
    holos = {} 
    count = 0
    zifs = mofgen(workdir, files)
    for zif in zifs:
        print(zif.name)
        count += 1
        holos[zif.name] = obtain_hologram(zif)
         
    hf = open('holograms.pkl', 'wb')
    pickle.dump(holos, hf)
    hf.close()

    #holo1.set_similarity_metric("tan_cont")
    #print(holo1 | holo2)
    timef=time()
    print("Walltime: %.2f seconds"%(timef-times))
if __name__=="__main__":
    main()
