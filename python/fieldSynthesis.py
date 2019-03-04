import numpy as np
import scipy as sci
import scipy.fftpack as ft
import scipy.signal as sig
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as colors
import sys
import argparse
import pdb

''' Field Synthesis
Python-based demonstration of Field Synthesis

supplementary material to:
    Universal Light-Sheet Generation with Field Synthesis
    Bo-Jui Chang, Mark Kittisopikul, Kevin M. Dean, Phillipe Roudot, Erik Welf and Reto Fiolka.

Mark Kittisopikul
Goldman Lab
Northwestern University

November 2018

Field Synthesis Demonstration -
Python code to demonstrate field synthesis light sheet microscopy
Copyright (C) 2019 Reto Fioka,
              University of Texas Southwestern Medical Center
Copyright (C) 2019 Mark Kittisopikul,
              Northwestern University
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

def createAnnulus(n=256, r=32, w=4):
    ''' createAnnulus - create a ring-like structure
    INPUT
    n - size of square array or vector
    r - radius of the ring
    w - width of the ring
    OUTPUT
    an array n x n
    '''
    if np.isscalar(n):
        v = np.arange(n)
        v = v - np.floor(n/2)
    else:
        v = n

    y,x = np.meshgrid(v,v)
    q = np.hypot(x,y)
    annulus = abs(q-r) < w
    return annulus

def createLatticeHat(n=256, r=32, w=4, s=24):
    ''' createLatticeHat - adapt annulus for a lattice configuration
    INPUT
    n,r,w - See createAnnulus
    s - location of the side lobes relative to the center
    OUTPUT
    an array n x n
    '''
    annulus = createAnnulus(n,r,w);
    latticeHat = np.zeros_like(annulus);
    center = n//2
    columns = (center-s,center,center+s)
    latticeHat[:,columns] = annulus[:,columns]
    return latticeHat;

def doConventionalScan(Fsqmod,Lsqmod):
    '''Simulate Conventional digital scanning / dithering
        INPUT
        F_sqmod - Square modulus of F at the front focal plane
        L_sqmod - Square modulus of L at the front focal plane
        OUTPUT
        scanned - Scanned (dithered) intensity of Fsqmod by Lsqmod
    '''
    # Manually scan by shifting Fsqmod and multiplying by Lsqmod
    scanned = np.zeros(Fsqmod.shape)
    center = Lsqmod.shape[1]//2

    for x in range(np.size(Fsqmod,1)):
        scanned = scanned + np.roll(Fsqmod,x-center,1)*Lsqmod[center,x]

    return scanned

def doConventionalScanHat(F_hat,L_hat):
    '''Simulate Conventional digital scanning / dithering from frequency space representations
       INPUT
       F_hat - Mask at back focal plane
       L_hat - Line scan profile in frequency space at the back focal plane
       OUTPUT
       scanned - Scanned (dithered) intensity of Fsqmod by Lsqmod at front focal plane
    '''
    F_hat = ft.ifftshift(F_hat)
    F = ft.ifft2(F_hat)
    F = ft.fftshift(F)
    # This is the illumination intensity pattern
    Fsqmod = np.real(F*np.conj(F))

    L_hat = ft.ifftshift(L_hat)
    L = ft.ifft2(L_hat)
    L = ft.fftshift(L)
    Lsqmod = L*np.conj(L)

    scanned = doConventionalScan(Fsqmod,Lsqmod)
    return scanned
    

def doFieldSynthesisLineScan(F_hat,L_hat):
    '''Simulate Field Synthesis Method
        INPUT
        F_hat - Frequency space representation of illumination pattern, mask at back focal plane
        L_hat - Line scan profile in frequency space at the back focal plane
        OUTPUT
        fieldSynthesis - Field synthesis construction by doing a line scan in the back focal plane
    '''
    # Do the Field Synthesis method of performing a line scan at the back focal plane
    fieldSynthesis = np.zeros_like(F_hat)

    max_T_a = 0

    for a in range(fieldSynthesis.shape[1]):
        # Instaneous scan in frequency space
        T_hat_a = F_hat * np.roll(L_hat,a-F_hat.shape[1]//2,1)
        # Instaneous scan in object space
        T_a = ft.fftshift( ft.fft2( ft.ifftshift(T_hat_a) ) )
        # Instaneous intensity
        T_a_sqmod = np.abs(T_a)**2
        # Track maximum value
        max_T_a = max(np.amax(T_a_sqmod),max_T_a)
        # Incoherent summing of the intensities
        fieldSynthesis = fieldSynthesis + T_a_sqmod

    return fieldSynthesis,max_T_a

def demoFieldSynthesis(animate=False,F_hat=None):
    '''Demonstrate Field Synthesis Method with Plots
        INPUT
         animate- boolean, if true, animate the figure
        OUTPUT
         None
    '''
    # plt.rc('text', usetex=True)
#    if animate:
#        plt.ion()
    # Style the figure
    # plt.style.use('dark_background');
    colormap = 'gnuplot2'

    fig, ax = plt.subplots(2,4,sharey=True,sharex=True,figsize=(16,9))

    # Create F, the illumination pattern
    if F_hat is None:
        F_hat = createAnnulus()
#        F_hat = createLatticeHat();
    F_hat = ft.ifftshift(F_hat)
    F = ft.ifft2(F_hat)
    F = ft.fftshift(F)
    # This is the illumination intensity pattern
    Fsqmod = np.real(F*np.conj(F))

    #plt.figure()
    #plt.title('F')
    #plt.imshow(Fsqmod)
    #plt.show(block=False)
    ax[0,0].imshow(Fsqmod)
    ax[0,0].set_title('|F(x,z)|^2')

    # Create L, the scan profile
    L = np.zeros_like(Fsqmod)
    center = L.shape[1]//2
    sigma = 30
    L[center,:] = norm.pdf(np.arange(-center,center),0,sigma)
    # L[L.shape[1]//2,:] = 1
    # The square modulus of L is the object space
    Lsqmod = L*np.conj(L)
    # This is the line scan profile used in Field Synthesis
    L_hat = ft.fftshift(ft.fft2(ft.ifftshift(L)))
    L_hat_abs = np.abs(L_hat)

    ax[0,1].imshow(L)
    ax[0,1].set_title('$ L(x)\delta(z) $')

    ax[0,2].imshow(Lsqmod)
    ax[0,2].set_title('$ |L(x)\delta(z)|^2 $')

    ax[0,3].imshow(np.abs(L_hat))
    ax[0,3].set_title('$\hat{L}(k_x) $')

    # Manually scan by shifting Fsqmod and multiplying by Lsqmod
    scanned = doConventionalScan(Fsqmod,Lsqmod)

    ax[1,0].imshow(scanned)
    ax[1,0].set_title('Scanned: $ \sum_{x\'} |F(x\',z)|^2|L(x-x\')|^2 $')

    # Manually scanning is a convolution operation
    # There are potentially boundary effects here
    convolved = sig.fftconvolve(Fsqmod,Lsqmod,'same')

    ax[1,1].imshow(convolved)
    ax[1,1].set_title('Conv: $ |F(x,z)|^2 ** |L(x)\delta(z)|^2 $')

    # This manual implementation of Fourier transform based convolution
    # actually does circular convolution
    convolvedft = ft.fftshift(ft.fft2(ft.ifft2(ft.ifftshift(Fsqmod)) *ft.ifft2(ft.ifftshift(Lsqmod))))
    convolvedft = np.real(convolvedft)

    ax[1,2].imshow(convolvedft)
    ax[1,2].set_title(r'Conv. FT: $ \mathcal{F}^{-1} \{ \mathcal{F}\{|F|^2\} \mathcal{F}\{|L(x)\delta(z)|^2\} \} $')

    # Do the Field Synthesis method of performing a line scan at the back focal plane
    fieldSynthesis,a = doFieldSynthesisLineScan(F_hat,L_hat)

    ax[1,3].imshow(fieldSynthesis)
    ax[1,3].set_title('FS: $ \sum_a |\mathcal{F}^{-1}\{ \hat{F}(k_x,k_z)\hat{L}(k_x-a) \}|^2 $')
#    pdb.set_trace()
    if animate:
        L2 = Lsqmod[Lsqmod.shape[0]//2,]
        maxL2 = np.max(L2)
        frames = np.flatnonzero(L2 > maxL2/100)
        #ani = animation.FuncAnimation(fig,scanUpdate,frames,fargs=(ax,frames[0],Fsqmod,L2),repeat=True,interval=100)

        A = ft.fftshift(F_hat)
        maxL_hat = np.max(L_hat_abs[0,])
        L_hat_idx = np.transpose(np.asmatrix(np.flatnonzero(L_hat_abs[0,] > maxL_hat/1e3))) - center
        fs_frames = np.asmatrix(np.flatnonzero(np.any(A,0)))
        fs_frames = fs_frames + L_hat_idx 
        fs_frames = fs_frames.flatten()
        fs_frames = np.unique(np.asarray(fs_frames)).flatten()
        #ani2 = animation.FuncAnimation(fig,fsUpdate,fs_frames,fargs=(ax,fs_frames[0],A,L_hat),repeat=True,interval=100)
        #nFrames = np.lcm(len(frames),len(fs_frames))
        #updateMethod = commonLCMUpdate
        nFrames = max(len(frames),len(fs_frames))
        updateMethod = commonMaxUpdate
        anilcm = animation.FuncAnimation(fig,updateMethod,nFrames, \
                fargs=(ax[(0,1),0],frames,Fsqmod,L2, \
                       ax[(0,1),3],fs_frames,A,L_hat), \
                repeat=True,interval=100)
        return fig,ax,anilcm

    return fig,ax

def retoMovie(animate=False,F_hat=None):
    '''Make a movie like Tweeted by @RetoPaul (Reto Fiolka'''
    # Style the figure
    # plt.style.use('dark_background');
    colormap = 'gnuplot2'

    fig, ax = plt.subplots(2,3,sharey=True,sharex=True,figsize=(16,9))

    # Create F, the illumination pattern
    if F_hat is None:
        F_hat = createAnnulus()
#        F_hat = createLatticeHat();
    F_hat = ft.ifftshift(F_hat)
    F = ft.ifft2(F_hat)
    F = ft.fftshift(F)
    # This is the illumination intensity pattern
    Fsqmod = np.real(F*np.conj(F))

    #plt.figure()
    #plt.title('F')
    #plt.imshow(Fsqmod)
    #plt.show(block=False)
    ax[0,0].imshow(ft.fftshift(F_hat))
    #ax[0,0].set_title('Scanned / Dithered',loc='left')
    ax[0,0].set_ylabel('Scanned / Dithered')

    ax[0,1].imshow(Fsqmod)
    #ax[0,1].set_title('|F(x,z)|^2')

    # Create L, the scan profile
    L = np.zeros_like(Fsqmod)
    center = L.shape[1]//2
    sigma = 30
    L[center,:] = norm.pdf(np.arange(-center,center),0,sigma)
    # L[L.shape[1]//2,:] = 1
    # The square modulus of L is the object space
    Lsqmod = L*np.conj(L)
    # This is the line scan profile used in Field Synthesis
    L_hat = ft.fftshift(ft.fft2(ft.ifftshift(L)))
    L_hat_abs = np.abs(L_hat)

    #ax[0,1].imshow(L)
    #ax[0,1].set_title('$ L(x)\delta(z) $')

    #ax[0,2].imshow(Lsqmod)
    #ax[0,2].set_title('$ |L(x)\delta(z)|^2 $')

    ax[1,0].imshow(np.abs(L_hat))
    #ax[1,0].set_title('Field Synthesis',loc='left')
    ax[1,0].set_ylabel('Field Synthesis')
    ax[1,0].set_xlabel('Fourier space')

    # Manually scan by shifting Fsqmod and multiplying by Lsqmod
    scanned = doConventionalScan(Fsqmod,Lsqmod)

    ax[0,2].imshow(scanned)
    #ax[0,2].set_title('Scanned: $ \sum_{x\'} |F(x\',z)|^2|L(x-x\')|^2 $')

    # Manually scanning is a convolution operation
    # There are potentially boundary effects here
    convolved = sig.fftconvolve(Fsqmod,Lsqmod,'same')

    # ax[1,1].imshow(convolved)
    # ax[1,1].set_title('Conv: $ |F(x,z)|^2 ** |L(x)\delta(z)|^2 $')

    # This manual implementation of Fourier transform based convolution
    # actually does circular convolution
    convolvedft = ft.fftshift(ft.fft2(ft.ifft2(ft.ifftshift(Fsqmod)) *ft.ifft2(ft.ifftshift(Lsqmod))))
    convolvedft = np.real(convolvedft)

    # ax[1,2].imshow(convolvedft)
    # ax[1,2].set_title(r'Conv. FT: $ \mathcal{F}^{-1} \{ \mathcal{F}\{|F|^2\} \mathcal{F}\{|L(x)\delta(z)|^2\} \} $')


    # Do the Field Synthesis method of performing a line scan at the back focal plane
    fieldSynthesis,max_T_a = doFieldSynthesisLineScan(F_hat,L_hat)

    T_a_hat = ft.fftshift(F_hat)*L_hat
    T_a_hat = ft.ifftshift(T_a_hat)
    T_a = ft.fftshift( ft.fft2(T_a_hat) )
    T_a = np.abs(T_a)**2

    ax[1,1].imshow(T_a,vmax=max_T_a)
    ax[1,1].set_xlabel('Real space\ninstantaneous')

    ax[1,2].imshow(fieldSynthesis)
    ax[1,2].set_xlabel('Real space\ntime averaged')
    #ax[1,2].set_title('FS: $ \sum_a |\mathcal{F}^{-1}\{ \hat{F}(k_x,k_z)\hat{L}(k_x-a) \}|^2 $')
#    pdb.set_trace()
    if animate:
        L2 = Lsqmod[Lsqmod.shape[0]//2,]
        maxL2 = np.max(L2)
        frames = np.flatnonzero(L2 > maxL2/100)
        #ani = animation.FuncAnimation(fig,scanUpdate,frames,fargs=(ax,frames[0],Fsqmod,L2),repeat=True,interval=100)

        A = ft.fftshift(F_hat)
        maxL_hat = np.max(L_hat_abs[0,])
        L_hat_idx = np.transpose(np.asmatrix(np.flatnonzero(L_hat_abs[0,] > maxL_hat/1e3))) - center
        fs_frames = np.asmatrix(np.flatnonzero(np.any(A,0)))
        fs_frames = fs_frames + L_hat_idx 
        fs_frames = fs_frames.flatten()
        fs_frames = np.unique(np.asarray(fs_frames)).flatten()
        #ani2 = animation.FuncAnimation(fig,fsUpdate,fs_frames,fargs=(ax,fs_frames[0],A,L_hat),repeat=True,interval=100)
        #nFrames = np.lcm(len(frames),len(fs_frames))
        #updateMethod = commonLCMUpdate
        nFrames = max(len(frames),len(fs_frames))
        updateMethod = commonMaxUpdate
        anilcm = animation.FuncAnimation(fig,updateMethod,nFrames, \
                fargs=(ax[0,(1,2)],frames,Fsqmod,L2, \
                       ax[1,(0,2,1)],fs_frames,A,L_hat), \
                repeat=True,interval=100)
    else:
        anilcm = None

    return fig,ax,anilcm




def commonLCMUpdate(frame,ax,frames,Fsqmod,L2,fs_ax,fsFrames,A,L_hat):
    '''commonUpdate updates animation for scanning and
    field synthesis for least common multiple frames
    '''
    f1 = frames[frame%len(frames)]
    f2 = fsFrames[frame%len(fsFrames)]
    scan_out  = scanUpdate(f1,ax,frames[0],Fsqmod,L2)
    fs_out = fsUpdate(f2,fs_ax,fsFrames[0],A,L_hat)
    return scan_out + fs_out

def commonMaxUpdate(frame,ax,frames,Fsqmod,L2,fs_ax,fsFrames,A,L_hat):
    '''commonUpdate updates animation for scanning and
    field synthesis for the greater number of frames
    '''
    scan_out = ()
    fs_out = ()
    if frame < len(frames):
        scan_out  = scanUpdate(frames[frame],ax,frames[0],Fsqmod,L2)
    if frame < len(fsFrames):
        fs_out = fsUpdate(fsFrames[frame],fs_ax,fsFrames[0],A,L_hat)
    return scan_out + fs_out

def scanUpdate(frame,ax,first,Fsqmod,L2):
    '''scanUpdate updates scanning for animation
    '''
    Fsqmod_im = ax[0].get_images()[0]
    im = ax[1].get_images()[0]
    global I
    if frame==first:
        I = np.zeros_like(Fsqmod)

    Fsqmod = np.roll(Fsqmod,frame-Fsqmod.shape[0]//2,1)
    I = I + L2[frame]*Fsqmod
    Fsqmod_im.set_array(Fsqmod*L2[frame]/np.max(L2))
    im.set_array(np.real(I))
    return im,Fsqmod_im


def fsUpdate(frame,ax,first,A,L_hat):
    '''fsUpdate updates field synthesis for animation
    '''
    global FS
    L_hat_im = ax[0].get_images()[0]
    fs_im = ax[1].get_images()[0]
    if frame==first:
        FS = np.zeros(A.shape)

    T_a_hat = A*np.roll(L_hat,frame-L_hat.shape[0]//2,1)
    L_hat_im.set_array(np.abs(T_a_hat))
    T_a_hat = ft.ifftshift(T_a_hat)
    T_a = ft.fftshift( ft.fft2(T_a_hat) )
    T_a = np.abs(T_a)**2
    FS = FS + T_a
    fs_im.set_array(FS)
    if len(ax) > 2:
        ta_im = ax[2].get_images()[0]
        ta_im.set_array(T_a)
        return L_hat_im,fs_im,ta_im
    else:
        return L_hat_im,fs_im

def applyRetoStyle():
    '''applyRetoStyle uses white text on a black background
    and does not show any frames, ticks, or labels'''
    plt.style.use('dark_background');
    mpl.rcParams['axes.edgecolor'] = 'black';
    mpl.rcParams['xtick.color'] = 'black';
    mpl.rcParams['ytick.color'] = 'black';
    mpl.rcParams['axes.labelsize'] = 'large';
    mpl.rcParams['image.cmap'] = 'gnuplot2'
#    mpl.rcParams['image.interpolation'] = 'bicubic';




if __name__ == "__main__":
    masks = {'bessel':createAnnulus,
            'b':createAnnulus,
            'hexagonal':createLatticeHat,
            'h':createLatticeHat,
            'square':lambda:createLatticeHat(s=30),
            's':lambda:createLatticeHat(s=30)}

    parser = argparse.ArgumentParser(description='Demonstrate Field Synthesis Light-sheet Microscopy')
    parser.add_argument('--animate','-a',action='store_true',default=True)
    parser.add_argument('--no-animate','-na',dest='animate',action='store_false')
    parser.add_argument('--movie','-m')
    parser.add_argument('--show','-s',action='store_true',default=True)
    parser.add_argument('--no-show','-ns',dest='show',action='store_false')
    parser.add_argument('--mask',default='bessel',choices=masks)
    parser.add_argument('--zoom',default=1.0,type=float)
    parser.add_argument('--proof',action='store_true',default=True)
    parser.add_argument('--reto',action='store_true',default=False)
    parser.add_argument('--anti',action='store_true',default=True)
    args = parser.parse_args()

    #pdb.set_trace()

    mask = masks[args.mask]()

    if args.anti:
        mask = mask.astype('float')
        center = mask.shape[0]//2
        mask[center+1:,] = -mask[center+1:,]

    # pdb.set_trace()
 
    if args.reto:
        applyRetoStyle()
        fig,ax,anilcm = retoMovie(args.animate,mask)
        if args.anti:
            darkcm = colors.ListedColormap([1,1,1,2] - cm.get_cmap('PiYG')(np.linspace(0,1,256)))
            ax[0,0].get_images()[0].set_cmap(darkcm)
    elif args.proof:
        fig,ax,anilcm = demoFieldSynthesis(args.animate,mask)

    if args.zoom > 1:
        xmin,xmax,ymin,ymax = plt.axis()
        xr = xmax-xmin
        yr = ymax-ymin
        zf = 0.5-0.5/args.zoom
        plt.axis((xmin+xr*zf,xmax-xr*zf,ymin+yr*zf,ymax-yr*zf))

    #for axx in ax.flatten():
    #    plt.setp(axx.spines.values(),color='black')
    #    plt.setp([axx.get_xticklines(), axx.get_yticklines()],color='black')
    #    plt.setp([axx.get_xticklabels(), axx.get_yticklabels()],color='black')

    if args.movie and anilcm is not None:
        anilcm.save(args.movie)

    if args.show:
        plt.show()
        plt.pause(0.0001)
