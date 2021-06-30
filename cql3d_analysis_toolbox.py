# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:33:48 2021

@author: kunalsanwalka

This program contains a list of functions used to extract useful information
from the netCDF4 files output by CQL3D.

Most functions are written to be standalone, requiring only the location of the
file. However, each function has its own docstring and one should refer to that
to understand function behaviour.

Packages needed to run functions in this file-
1. numpy as np
2. netCDF4 as nc
3. matplotlib.pyplot as plt
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# =============================================================================
# Destination of the plots
# =============================================================================
plotDest='./plots/'

def zero_d_parameters(filename):
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    
    
    return

def species_labels(filename):
    """
    This function generates an array with the labels for each 'general' species
    That is, a species whos distribution function has been explicitly 
    calculated by CQL3D.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.

    Returns
    -------
    speciesLabels : array
        Labels of all the general species.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Name of each species and specification (general, maxwellian etc.)
    kspeci=np.array(ds['kspeci'][:])
    
    # =========================================================================
    # Generate the species labels
    # =========================================================================
    
    #Array to store species labels
    speciesLabels=[]
    
    #Get a slice of kspeci which just has the label and type
    kspeciSlice=kspeci[:,:,0]
    #Append correct names to the labelling array
    for i in range(len(kspeciSlice)):
        if kspeciSlice[i,1]==b'g':
            
            #Label in CQL3D
            cqlLabel=kspeciSlice[i,0].decode('utf-8')
            
            #Come up with a nicer label
            niceLabel=''
            if cqlLabel=='d' or cqlLabel=='D' or cqlLabel=='Deuterium' or cqlLabel=='deuterium':
                niceLabel='D'
            elif cqlLabel=='t' or cqlLabel=='T' or cqlLabel=='Tritium' or cqlLabel=='tritium':
                niceLabel='T'
                
            #Add it to the array
            speciesLabels.append(niceLabel)
    
    return speciesLabels

def dist_func(filename,makeplot=False,saveplot=False,fluxsurfplot=0,species=0):
    """
    This function returns the plasma distribution function along with the
    associated velocity coordinate arrays. If there are multiple general
    species in the CQL3D run, the 'species' keyword picks the species being
    plotted and returned by this function.
    
    The distribution function as output by CQL3D is of the form-
    f(rdim,xdim,ydim)
    
    Here-
    rdim = Flux surface
    xdim = Momentum-per-mass (=x)
    ydim = Pitch angle (=y)
    
    To convert this into f(vPar,vPerp) we need to construct the vPar and vPerp
    arrays. This is done by decomposing the momentum (~velocity) into its
    parallel and perpendicular components based on the pitch angle.
    
    All three output arrays are 3D with the first index used to indicate the
    flux surface.
    
    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
    fluxsurfplot : int
        Flux surface number to be plotted (0=innermost flux surface).
    species : int
        Index of species.

    Returns
    -------
    f : np.array
        Distribution function. 
        It has the form- f(flux surface,v_par,v_perp)
    vPar : np.array
        Parallel velocity (with respect to the magnetic field).
        It has the form- vPar(flux surface,pitch angle,momentum-per-mass)
    vPerp : np.array
        Perpendicular velocity (with respect to the magnetic field).
        It has the form- vPerp(flux surface,pitch angle,momentum-per-mass)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Distribution function
    f=np.array(ds['f'][:])
    
    #Pitch angle array
    y=np.array(ds['y'][:])

    #Maximum pitch angle dimension (=ydim)
    iy=int(ds['iy'][:])
    
    #Normalized momentum-per-mass array
    x=np.array(ds['x'][:])
    
    #Momentum-per-mass dimension (=xdim)
    jx=int(ds['jx'][:])
    
    #Velocity normalization factor
    vnorm=int(ds['vnorm'][:])
    
    #Number of radial surface bins (=rdim)
    lrz=int(ds['lrz'][:])
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #Species Labels
    speciesLabels=species_labels(filename)
    
    # =========================================================================
    # Create the vperp and vpar arrays
    # =========================================================================
    
    #Create the variables
    vPerp=np.zeros((lrz,iy,jx))
    vPar=np.zeros((lrz,iy,jx))
    
    #Calculate the velocities based on the magnitude and pitch angle for each
    #flux surface
    for k in range(0,lrz):
        for i in range(0,iy):
            for j in range(0,jx):
                vPerp[k,j,i]=vnorm*x[j]*np.sin(y[k,i])
                vPar[k,j,i]=vnorm*x[j]*np.cos(y[k,i])
    
    #Convert velocity to m/s from cm/s
    vPar/=100
    vPerp/=100
    
    # =========================================================================
    # Clean up distribution function
    # =========================================================================
    
    #Remove all nan values (set them to 0)
    f=np.nan_to_num(f)
    
    #Set all values at or below 0 to 10e-5 (helps with taking the log)
    f[f<=0]=1e-5
    
    # =========================================================================
    # Check if the distribution function has multiple species
    # =========================================================================
    
    multiSpecies=False
    
    if ngen>1:
        
        multiSpecies=True
        
        #Get the f for the right species (else f has the wrong shape)
        f=f[species]
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_dist_func_fluxsurf_'+str(fluxsurfplot)+'.svg'
        if multiSpecies:
            savename=ncName+'_dist_func_fluxsurf_'+str(fluxsurfplot)+'_species_'+speciesLabels[species]+'.svg'
        
        #Represent the r/a value in scientific notation
        ryaSciNot="{:.2e}".format(rya[fluxsurfplot])
        
        #Convert data to log
        logData=np.log10(f[fluxsurfplot])
        
        #Maximum of the distribution
        maxDist=np.max(logData)
        minDist=maxDist-15
        
        #Create the plot
        fig=plt.figure(figsize=(21,8))
        ax=fig.add_subplot(111)
        pltobj=ax.contourf(vPar[fluxsurfplot],vPerp[fluxsurfplot],logData,levels=np.linspace(minDist,maxDist,30))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel(r'$v_{||}$ [m/s]')
        ax.set_xlim(-8e6,8e6)
        ax.set_xticks(np.linspace(-8e6,8e6,17))
        ax.set_ylabel(r'$v_{\perp}$ [m/s]')
        ax.set_ylim(0,8e6)
        ax.set_title('Distribution Function (r/a = '+ryaSciNot+')')
        ax.grid(True)
        cbar=fig.colorbar(pltobj)
        cbar.set_label(r'log$_{10}$(v$^{-3}$)')
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
        plt.show()
    
    return f,vPar,vPerp
    
def ion_dens(filename,makeplot=False,saveplot=False,efastd=6,species=0):
    """
    This function returns the ion densities along with the associated 
    coordinate arrays.
    
    CQL3D does not output the ion densities directly. They are calculated by
    taking an integral of the distribution function over velocity space.
    
    Here, the ion densities are defined by-
    
    Fast ions = >6keV
    Warm tons = <6keV
    Total ions = Warm ions + Fast ions
    
    This threshold can be changed by altering the 'eFastd' variable.
    
    NOTE: This function was originally written in Fortran, then converted
          to IDL and is finally in Python. A lot of optimizations can be made 
          to this code that are Python specific.

    Parameters
    ----------
    filename : string
        Location of the CQl3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
    efastd : float
        Boundary between warm and fast ions (keV).
    species : int
        Index of species.

    Returns
    -------
    ndwarmz : np.array
        Warm ion density function.
        It has the form- ndwarmz(radial position,z position)
    ndfz : np.array
        Fast ion density function.
        It has the form- ndfz(radial position,z position)
    ndtotz : np.array
        Total ion density function.
        It has the form- ndtotz(radial position,z position)
    solrz : np.array
        Radial position.
        It has the form- solrz(radial position,z position)
    solzz : np.array
        Z position.
        It has the form- solzz(radial position,z position)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Major radius of z points (=r)
    solrz=np.array(ds['solrz'][:])
    
    #Height of z points (=z)
    solzz=np.array(ds['solzz'][:])
    
    #Dimension of z-grid along B
    lz=int(ds['lz'][:])
    
    #Number of radial surface bins (=rdim)
    lrz=int(ds['lrz'][:])
    
    #Distribution function
    f=np.array(ds['f'][:])
    
    #Pitch angle array
    y=np.array(ds['y'][:])
    
    #Maximum pitch angle dimension (=ydim)
    iy=int(ds['iy'][:])
    
    #Normalized momentum-per-mass array
    x=np.array(ds['x'][:])
    
    #Momentum-per-mass dimension (=xdim)
    jx=int(ds['jx'][:])
    
    #dx centered on x-mesh points
    dx=np.array(ds['dx'][:])
    
    #Velocity normalization factor
    vnorm=int(ds['vnorm'][:])
    
    #Normalized magnetic field strength (B(z)/B(z=0))
    bbpsi=np.array(ds['bbpsi'][:])
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #Species Labels
    speciesLabels=species_labels(filename)
    
    # =========================================================================
    # Clean up distribution function
    # =========================================================================
    
    #Remove all nan values (set them to 0)
    f=np.nan_to_num(f)
    
    #Set all values at or below 0 to 10e-5 (helps with taking the log)
    f[f<=0]=1e-5
    
    # =========================================================================
    # Check if the distribution function has multiple species
    # =========================================================================
    
    multiSpecies=False
    
    if ngen>1:
        
        multiSpecies=True
        
        #Get the f for the right species (else f has the wrong shape)
        f=f[species]
    
    # =========================================================================
    # Create the ion density arrays
    # =========================================================================
    
    #Pitch angles from one central radial point
    pitchAngleArr=y[0,:]
    
    #Pitch angle step size
    dtheta=np.max(pitchAngleArr)/len(pitchAngleArr)
    dthetad2=0.5*dtheta
    
    #Create the theta arrays
    theta0=np.zeros(iy) #Uniformly spaced theta array
    ctheta=np.zeros(iy) #cos of theta0
    stheta=np.zeros(iy) #sin of theta0
    #TODO- Figure out what these are for. They look like various integrands
    theta1=np.zeros(iy)
    theta2=np.zeros(iy)
    theta3=np.zeros(iy)
    
    #Define the values of the theta arrays
    for i in range(0,iy):
        theta0[i]=dthetad2+i*dtheta
        stheta[i]=np.sin(theta0[i])
        ctheta[i]=np.cos(theta0[i])
        theta1[i]=2*np.pi*stheta[i]*dtheta
        theta2[i]=theta1[i]*(ctheta[i]**2)
        theta3[i]=np.pi*(stheta[i]**3)*dtheta
        
    #Create the x location arrays
    #TODO- Figure out what these are for. They look like various integrands
    xloc1=np.zeros(jx)
    xloc2=np.zeros(jx)
    
    #Define the xloc arrays
    for i in range(0,jx):
        xloc1[i]=(x[i]**2)*dx[i]
        xloc2[i]=(vnorm**2)*(x[i]**2)*xloc1[i]
        
    #Create the cosz and sinz arrays
    cosz=np.zeros((iy,lz,lrz))
    bsinz=np.zeros((iy,lz,lrz))
    
    #Define cosz and sinz
    for ilr in range(0,lrz): #flux surfaces
        for ilz in range(0,lz): #z positions
            for i in range(0,iy): #pitch angles
                sign=-1    
                if y[ilr,i]<=(np.pi/2):
                    sign=1.0
                else:
                    sign=-1.0
                if (1-bbpsi[ilr,ilz]*np.sin(y[ilr,i])**2)>0:
                    cosz[i,ilz,ilr]=sign*np.sqrt(1-bbpsi[ilr,ilz]*np.sin(y[ilr,i])**2)
                bsinz[i,ilz,ilr]=np.sqrt(1-cosz[i,ilz,ilr]**2)
    
    #Create itheta
    #TODO- What is itheta?
    itheta=np.zeros((iy,lz,lrz))
    
    #Define itheta
    for lr in range(0,lrz):
        for l in range(0,lz):
            for i in range(0,int(iy/2)):
                tempvalArr=np.where(bsinz[0:int(iy/2),l,lr]>=stheta[i])
                if np.size(tempvalArr)==0:
                    tempval=0
                else:
                    tempval=np.min(tempvalArr)
                #Check if tempval is larger than iy/2
                if tempval>(iy/2):
                    itheta[i,l,lr]=int(iy/2)
                else:
                    itheta[i,l,lr]=tempval
                #Make itheta symmetric
                itheta[iy-i-1,l,lr]=itheta[i,l,lr]
    
    #Create the ion density arrays
    ndfz=np.zeros((lz,lrz)) #Fast ions
    ndwarmz=np.zeros((lz,lrz)) #Warm ions
    ndtotz=np.zeros((lz,lrz)) #Total ions
    
    #Species atomic number (assume D for single species)
    anumd=2
    if speciesLabels[species]=='D':
        anumd=2
    elif speciesLabels[species]=='T':
        anumd=3
        
    #Velocity of the fast ions
    vfastd=np.sqrt(2*efastd*1000/(anumd*938e6))*3e10
    #Array with indices where velocity is greater than vfastd
    fastArr=np.where(vnorm*x>=vfastd)
    #Minimum index
    jfast_mind=np.min(fastArr)
    
    #Calculate the fast ion density
    for ilr in range(0,lrz): #flux surfaces
        for ilz in range(0,lz): #z positions
            for ij in range(0,jx): #energy bins
                for i in range(0,iy): #pitch angles
                    icl=int(ilr/lrz*lrz)
                    ithetahere=int(itheta[i,ilz,ilr])
                    ndtotz[ilz,ilr]+=theta1[i]*xloc1[ij]*f[icl,ij,ithetahere]
                    if ij>=jfast_mind:
                        ndfz[ilz,ilr]+=theta1[i]*xloc1[ij]*f[icl,ij,ithetahere]
                    else:
                        ndwarmz[ilz,ilr]+=theta1[i]*xloc1[ij]*f[icl,ij,ithetahere]
    
    #Transpose the density arrays to match the indexing convention of Python. 
    #ndfz,ndwarmz and ndtotz use the same indexing convention as IDL. Since 
    #solrz and solzz follow the Python convention, we need to make sure the 
    #density arrays are consistent with solrz and solzz
    ndfz=np.transpose(ndfz)
    ndtotz=np.transpose(ndtotz)
    ndwarmz=np.transpose(ndwarmz)
    
    #Convert solrz and solzz from cm to m
    solrz/=100
    solzz/=100
    
    #Convert density from 1/cm^3 to 1/m^3
    ndwarmz*=1e6
    ndfz*=1e6
    ndtotz*=1e6
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        #Add suffix for all the plots
        savenameFast=ncName+'_fast_ion_dens.pdf'
        savenameTot=ncName+'_total_ion_dens.pdf'
        savenameWarm=ncName+'_warm_ion_dens.pdf'
        if multiSpecies:
            savenameFast=ncName+'_fast_ion_dens_species_'+speciesLabels[species]+'.pdf'
            savenameTot=ncName+'_total_ion_dens_species_'+speciesLabels[species]+'.pdf'
            savenameWarm=ncName+'_warm_ion_dens_species_'+speciesLabels[species]+'.pdf'
        
        #Normalize all plots with respect to each other
        maxDens=np.max(ndtotz)
        
        # =====================================================================
        # Fast ion density
        # =====================================================================
        
        fig1=plt.figure(figsize=(20,8))
        ax1=fig1.add_subplot(111)
        pltobj=ax1.contourf(solzz,solrz,ndfz,levels=np.linspace(0,maxDens,50))
        ax1.contour(pltobj,colors='black')
        ax1.set_xlabel('Z [m]')
        ax1.set_ylabel('R [m]')
        ax1.set_title('Fast Ion Density (>'+str(efastd)+'keV)')
        ax1.grid(True)
        cbar1=fig1.colorbar(pltobj)
        cbar1.set_label(r'Density [m$^{-3}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenameFast,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Warm ion density
        # =====================================================================
        
        fig2=plt.figure(figsize=(20,8))
        ax2=fig2.add_subplot(111)
        pltobj=ax2.contourf(solzz,solrz,ndwarmz,levels=np.linspace(0,maxDens,50))
        ax2.contour(pltobj,colors='black')
        ax2.set_xlabel('Z [m]')
        ax2.set_ylabel('R [m]')
        ax2.set_title('Warm Ion Density (<'+str(efastd)+'keV)')
        ax2.grid(True)
        cbar2=fig2.colorbar(pltobj)
        cbar2.set_label(r'Density [m$^{-3}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenameWarm,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Total ion density
        # =====================================================================
        
        fig3=plt.figure(figsize=(20,8))
        ax3=fig3.add_subplot(111)
        pltobj=ax3.contourf(solzz,solrz,ndtotz,levels=np.linspace(0,maxDens,50))
        ax3.contour(pltobj,colors='black')
        ax3.set_xlabel('Z [m]')
        ax3.set_ylabel('R [m]')
        ax3.set_title('Total Ion Density')
        ax3.grid(True)
        cbar3=fig3.colorbar(pltobj)
        cbar3.set_label(r'Density [m$^{-3}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenameTot,bbox_inches='tight')
        plt.show()
    
    return ndwarmz,ndfz,ndtotz,solrz,solzz
    
def pressure(filename,makeplot=False,saveplot=False,species=0):
    """
    This function returns the pressures for each species along with the 
    associated coordinate arrays.
    
    CQL3D does not output the pressures directly. They are calculated by
    taking an integral of the distribution function over velocity space.
    
    NOTE: This function was originally written in Fortran, then converted
          to IDL and is finally in Python. A lot of optimizations can be made 
          to this code that are Python specific.

    Parameters
    ----------
    filename : string
        Location of the CQl3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
    species : int
        Index of species.

    Returns
    -------
    pressparz_d : np.array
        Parallel pressure.
        It has the form- pressparz_d(radial position,z position)
    pressprpz_d : np.array
        Perpendicular pressure.
        It has the form- pressprpz_d(radial position,z position)
    pressz_d : np.array
        Total pressure.
        It has the form- pressz_d(radial position,z position)
    solrz : np.array
        Radial position.
        It has the form- solrz(radial position,z position)
    solzz : np.array
        Z position.
        It has the form- solzz(radial position,z position)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Major radius of z points (=r)
    solrz=np.array(ds['solrz'][:])
    
    #Height of z points (=z)
    solzz=np.array(ds['solzz'][:])
    
    #Dimension of z-grid along B
    lz=int(ds['lz'][:])
    
    #Number of radial surface bins (=rdim)
    lrz=int(ds['lrz'][:])
    
    #Distribution function
    f=np.array(ds['f'][:])
    
    #Pitch angle array
    y=np.array(ds['y'][:])
    
    #Maximum pitch angle dimension (=ydim)
    iy=int(ds['iy'][:])
    
    #Normalized momentum-per-mass array
    x=np.array(ds['x'][:])
    
    #Momentum-per-mass dimension (=xdim)
    jx=int(ds['jx'][:])
    
    #dx centered on x-mesh points
    dx=np.array(ds['dx'][:])
    
    #Velocity normalization factor
    vnorm=int(ds['vnorm'][:])
    
    #Normalized magnetic field strength (B(z)/B(z=0))
    bbpsi=np.array(ds['bbpsi'][:])
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #Species Labels
    speciesLabels=species_labels(filename)
    
    # =========================================================================
    # Clean up distribution function
    # =========================================================================
    
    #Remove all nan values (set them to 0)
    f=np.nan_to_num(f)
    
    #Set all values at or below 0 to 10e-5 (helps with taking the log)
    f[f<=0]=1e-5
    
    # =========================================================================
    # Check if the distribution function has multiple species
    # =========================================================================
    
    multiSpecies=False
    
    if ngen>1:
        
        multiSpecies=True
        
        #Get the f for the right species (else f has the wrong shape)
        f=f[species]
    
    # =========================================================================
    # Calculate the pressure profiles
    # =========================================================================
    
    #Pitch angles from one central radial point
    pitchAngleArr=y[0,:]
    
    #Pitch angle step size
    dtheta=np.max(pitchAngleArr)/len(pitchAngleArr)
    dthetad2=0.5*dtheta
    
    #Create the theta arrays
    theta0=np.zeros(iy) #Uniformly spaced theta array
    ctheta=np.zeros(iy) #cos of theta0
    stheta=np.zeros(iy) #sin of theta0
    #TODO- Figure out what these are for. They look like various integrands
    theta1=np.zeros(iy)
    theta2=np.zeros(iy)
    theta3=np.zeros(iy)
    
    #Define the values of the theta arrays
    for i in range(0,iy):
        theta0[i]=dthetad2+i*dtheta
        stheta[i]=np.sin(theta0[i])
        ctheta[i]=np.cos(theta0[i])
        theta1[i]=2*np.pi*stheta[i]*dtheta
        theta2[i]=theta1[i]*(ctheta[i]**2)
        theta3[i]=np.pi*(stheta[i]**3)*dtheta
        
    #Create the x location arrays
    #TODO- Figure out what these are for. They look like various integrands
    xloc1=np.zeros(jx)
    xloc2=np.zeros(jx)
    
    #Define the xloc arrays
    for i in range(0,jx):
        xloc1[i]=(x[i]**2)*dx[i]
        xloc2[i]=(vnorm**2)*(x[i]**2)*xloc1[i]
        
    #Create the cosz and sinz arrays
    cosz=np.zeros((iy,lz,lrz))
    bsinz=np.zeros((iy,lz,lrz))
    
    #Define cosz and sinz
    for ilr in range(0,lrz): #flux surfaces
        for ilz in range(0,lz): #z positions
            for i in range(0,iy): #pitch angles
                sign=-1    
                if y[ilr,i]<=(np.pi/2):
                    sign=1.0
                else:
                    sign=-1.0
                if (1-bbpsi[ilr,ilz]*np.sin(y[ilr,i])**2)>0:
                    cosz[i,ilz,ilr]=sign*np.sqrt(1-bbpsi[ilr,ilz]*np.sin(y[ilr,i])**2)
                bsinz[i,ilz,ilr]=np.sqrt(1-cosz[i,ilz,ilr]**2)
    
    #Create itheta
    #TODO- What is itheta?
    itheta=np.zeros((iy,lz,lrz))
    
    #Define itheta
    for lr in range(0,lrz):
        for l in range(0,lz):
            for i in range(0,int(iy/2)):
                tempvalArr=np.where(bsinz[0:int(iy/2),l,lr]>=stheta[i])
                if np.size(tempvalArr)==0:
                    tempval=0
                else:
                    tempval=np.min(tempvalArr)
                #Check if tempval is larger than iy/2
                if tempval>(iy/2):
                    itheta[i,l,lr]=int(iy/2)
                else:
                    itheta[i,l,lr]=tempval
                #Make itheta symmetric
                itheta[iy-i-1,l,lr]=itheta[i,l,lr]
                
    #Create the pressure profile arrays
    pressparz_d=np.zeros((lz,lrz)) #Parallel pressure
    pressprpz_d=np.zeros((lz,lrz)) #Perpendicular pressure
    
    #Mass of the species (assume D for single species)
    anumd=2
    if speciesLabels[species]=='D':
        anumd=2
    elif speciesLabels[species]=='T':
        anumd=3
    massSpec=anumd*1.67e-24
    
    #Calculate the pressure profiles
    for ilr in range(0,lrz): #flux surfaces
        for ilz in range(0,lz): #z positions
            for i in range(0,iy): #pitch angles
                for ij in range(0,jx): #energy bins
                    icl=int(ilr/lrz*lrz)
                    ithetahere=int(itheta[i,ilz,ilr])
                    pressparz_d[ilz,ilr]+=massSpec*theta2[i]*xloc2[ij]*f[icl,ij,ithetahere]
                    pressprpz_d[ilz,ilr]+=massSpec*theta3[i]*xloc2[ij]*f[icl,ij,ithetahere]
                    
    #Transpose the pressure arrays to match the indexing convention of Python. 
    #pressprpz_d and pressprpz_d use the same indexing convention as IDL. Since 
    #solrz and solzz follow the Python convention, we need to make sure the 
    #pressure arrays are consistent with solrz and solzz
    pressparz_d=np.transpose(pressparz_d)
    pressprpz_d=np.transpose(pressprpz_d)
    
    #Total pressure
    pressz_d=(pressparz_d+2*pressprpz_d)/3
    
    #Convert solrz and solzz to m from cm
    solrz/=100
    solzz/=100
    
    #Convert pressures to pascals from baryes
    pressparz_d/=10
    pressprpz_d/=10
    pressz_d/=10
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savenamePar=ncName+'_par_pressure.pdf'
        savenamePerp=ncName+'_perp_pressure.pdf'
        savenameTot=ncName+'_tot_pressure.pdf'
        if multiSpecies:
            savenamePar=ncName+'_par_pressure_'+speciesLabels[species]+'.pdf'
            savenamePerp=ncName+'_perp_pressure_'+speciesLabels[species]+'.pdf'
            savenameTot=ncName+'_tot_pressure_'+speciesLabels[species]+'.pdf'
        
        #Normalize all plots with respect to each other
        maxPressure=np.max([np.max(pressz_d),np.max(pressparz_d),np.max(pressprpz_d)])
        
        # =====================================================================
        # Parallel Pressure
        # =====================================================================
        
        fig1=plt.figure(figsize=(20,8))
        ax=fig1.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,pressparz_d,levels=np.linspace(0,maxPressure,50))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Parallel Pressure')
        ax.grid(True)
        cbar1=fig1.colorbar(pltobj)
        cbar1.set_label(r'Pascals [N/m$^{-2}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenamePar,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Perpendicular Pressure
        # =====================================================================
        
        fig2=plt.figure(figsize=(20,8))
        ax=fig2.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,pressprpz_d,levels=np.linspace(0,maxPressure,50))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Perpendicular Pressure')
        ax.grid(True)
        cbar2=fig2.colorbar(pltobj)
        cbar2.set_label(r'Pascals [N/m$^{-2}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenamePerp,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Total Pressure
        # =====================================================================
        
        fig3=plt.figure(figsize=(20,8))
        ax=fig3.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,pressz_d,levels=np.linspace(0,maxPressure,50))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Total Pressure')
        ax.grid(True)
        cbar3=fig3.colorbar(pltobj)
        cbar3.set_label(r'Pascals [N/m$^{-2}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenameTot,bbox_inches='tight')
        plt.show()
    
    return pressparz_d,pressprpz_d,pressz_d,solrz,solzz

def total_pressure(filename,makeplot=False,saveplot=False):
    """
    This function returns the total plasma pressure along with the associated
    coordinate arrays.

    Parameters
    ----------
    filename : string
        Location of the CQl3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    pressparz : np.array
        Parallel pressure.
        It has the form- pressparz_d(radial position,z position)
    pressprpz : np.array
        Perpendicular pressure.
        It has the form- pressprpz_d(radial position,z position)
    pressz : np.array
        Total pressure.
        It has the form- pressz_d(radial position,z position)
    solrz : np.array
        Radial position.
        It has the form- solrz(radial position,z position)
    solzz : np.array
        Z position.
        It has the form- solzz(radial position,z position)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    # =========================================================================
    # Calculate the total pressure
    # =========================================================================
    
    #Initialize the arrays
    pressparz,pressprpz,pressz,solrz,solzz=pressure(filename)
    
    #Use the standard function if there is only 1 species
    if ngen<=1:
        return pressure(filename,makeplot=makeplot,saveplot=saveplot)
    
    #Add the rest of the species
    for i in range(1,ngen):
        pressparzNew,pressprpzNew,presszNew,solrz,solzz=pressure(filename,species=i)
        pressparz+=pressparzNew
        pressprpz+=pressprpzNew
        pressz+=presszNew
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savenamePar=ncName+'_par_pressure.pdf'
        savenamePerp=ncName+'_perp_pressure.pdf'
        savenameTot=ncName+'_tot_pressure.pdf'
        
        #Normalize all plots with respect to each other
        maxPressure=np.max([np.max(pressz),np.max(pressparz),np.max(pressprpz)])
        
        # =====================================================================
        # Parallel Pressure
        # =====================================================================
        
        fig1=plt.figure(figsize=(20,8))
        ax=fig1.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,pressparz,levels=np.linspace(0,maxPressure,50))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Parallel Pressure')
        ax.grid(True)
        cbar1=fig1.colorbar(pltobj)
        cbar1.set_label(r'Pascals [N/m$^{-2}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenamePar,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Perpendicular Pressure
        # =====================================================================
        
        fig2=plt.figure(figsize=(20,8))
        ax=fig2.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,pressprpz,levels=np.linspace(0,maxPressure,50))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Perpendicular Pressure')
        ax.grid(True)
        cbar2=fig2.colorbar(pltobj)
        cbar2.set_label(r'Pascals [N/m$^{-2}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenamePerp,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Total Pressure
        # =====================================================================
        
        fig3=plt.figure(figsize=(20,8))
        ax=fig3.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,pressz,levels=np.linspace(0,maxPressure,50))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Total Pressure')
        ax.grid(True)
        cbar3=fig3.colorbar(pltobj)
        cbar3.set_label(r'Pascals [N/m$^{-2}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenameTot,bbox_inches='tight')
        plt.show()
    
    return pressparz,pressprpz,pressz,solrz,solzz

def beta(filename,makeplot=False,saveplot=False):
    """
    This function returns the plasma beta along with the associated coordinate 
    arrays.
    
    CQL3D does not output the pressures directly. They are calculated by
    taking moments of the distribution function over velocity space.
    
    NOTE: This function was originally written in Fortran, then converted
          to IDL and is finally in Python. A lot of optimizations can be made 
          to this code that are Python specific.

    Parameters
    ----------
    filename : string
        Location of the CQl3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    beta_z : np.array
        Plasma beta as a function of position.
        It has the form- beta_z(radial position,z position)
    solrz : np.array
        Radial position.
        It has the form- solrz(radial position,z position)
    solzz : np.array
        Z position.
        It has the form- solzz(radial position,z position)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Dimension of z-grid along B
    lz=int(ds['lz'][:])
    
    #Number of radial surface bins (=rdim)
    lrz=int(ds['lrz'][:])
    
    #Normalized magnetic field strength (B(z)/B(z=0))
    bbpsi=np.array(ds['bbpsi'][:])
    
    #Minimum |B| on a flux surface
    bmidplane=np.array(ds['bmidplne'][:])
    
    # =========================================================================
    # Calculate the plasma beta
    # =========================================================================
    
    #Get the pressure profiles
    pressparz_d,pressprpz_d,pressz_d,solrz,solzz=total_pressure(filename)
    
    #Transpose to match the IDL code
    pressparz_d=np.transpose(pressparz_d)
    pressprpz_d=np.transpose(pressprpz_d)
    pressz_d=np.transpose(pressz_d)
    
    #Create the beta array
    beta_z=np.zeros((lz,lrz))
    
    #Create bzz (Magnetic field strength)
    bzz=np.zeros((lz,lrz))
    
    #Calculate bzz
    for i in range(0,lrz):
        bzz[:,i]=bbpsi[i,:]*bmidplane[i]
        
    #Calculate beta_z
    beta_z=8*np.pi*pressz_d/bzz**2
    
    #Transpose the beta array to match the indexing convention of Python. 
    #beta_z use the same indexing convention as IDL. Since solrz and solzz 
    #follow the Python convention, we need to make sure the beta array is
    #consistent with solrz and solzz
    beta_z=np.transpose(beta_z)
    
    #Multiply beta by 10 because the units in the above calculation are mixed
    beta_z*=10
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_beta.pdf'
        
        fig=plt.figure(figsize=(20,8))
        ax=fig.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,beta_z,levels=50)
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Plasma Beta')
        ax.grid(True)
        fig.colorbar(pltobj)
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
        plt.show()
    
    return beta_z,solrz,solzz

def axial_neutron_flux(filename,makeplot=False,saveplot=False):
    """
    This function returns the fustion neutron flux as a function of the axial
    coordinate z.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    flux_neutron_f : np.array
        Fusion neutron flux as a function of the axial coordinate z.
        Units- W/m**2/steradian
    z_fus : np.array
        Values of z associated with flux_neutron_f.
        Units- m
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Neutron flux as a function of z
    flux_neutron_f=np.array(ds['flux_neutron_f'][:])
    
    #z array associated with the neutron flux array
    z_fus=np.array(ds['z_fus'][:])
    
    # =========================================================================
    # Convert to SI units
    # =========================================================================
    
    #Convert the z array into meters
    z_fus/=100
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_fus_flux_axial_dependence.png'
        
        fig=plt.figure(figsize=(15,8))
        ax=fig.add_subplot(111)
        ax.plot(z_fus,flux_neutron_f)
        ax.set_xlabel('Z [m]')
        ax.set_ylabel(r'Fusion Neutron Flux [W/m$^2$sr]')
        ax.grid()
        ax.set_title('Fusion Neutron Flux')
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
        plt.show()
    
    return flux_neutron_f,z_fus

def total_fusion_power(filename):
    """
    This function returns the total fusion power for 4 different reactions.
    
    The order of the reactions in the array that is returned is-
    1. D + T --> n + 4He
    2. D + 3He --> p + 4He
    3. D + D --> n + 3He
    4. D + D --> p + T

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.

    Returns
    -------
    fuspwrvt : np.array
        Array with the various reaction power outputs.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Total fusion power for 4 different reactions
    fuspwrvt=np.array(ds['fuspwrvt'][:])
    
    return fuspwrvt

def fusion_rx_rate(filename,makeplot=False,saveplot=False):
    """
    This function returns the reaction rate a function of time for 4 different
    reactions.
    
    The order of the reactions in the array that is returned is-
    1. D + T --> n + 4He
    2. D + 3He --> p + 4He
    3. D + D --> n + 3He
    4. D + D --> p + T

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    sigfft : np.array
        2D array with the reaction rates. Major axis selects for the reaction.
    time : np.array
        Corresponding time array.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Fusion reaction rate as a function of time
    sigftt=np.array(ds['sigftt'][:])
    
    #Corresponding time axis
    time=np.array(ds['time'][:])
    
    # =========================================================================
    # Transpose to allow easy selection of different reactions
    # =========================================================================
    
    sigftt=np.transpose(sigftt)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_fus_rx_rate.png'
        
        fig=plt.figure(figsize=(8,8))
        ax=fig.add_subplot(111)
        ax.plot(time,sigftt[0],label=r'D + T --> n + $^4$He')
        ax.plot(time,sigftt[1],label=r'D + $^3$He --> p + $^4$He')
        ax.plot(time,sigftt[2],label=r'D + D --> n + $^3$He')
        ax.plot(time,sigftt[3],label='D + D --> p + T')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'Fusion Reaction Rate [s$^{-1}$]')
        ax.grid()
        ax.set_title('Fusion Reaction Rate')
        ax.legend(bbox_to_anchor=(1,1))
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
        plt.show()
    
    return sigftt,time

def plot_dist_funcs(filename,saveplot=False,species=0):
    """
    This function plots the distribution function on all flux surfaces. If
    there are multiple general species in the run, the 'species' keyword
    selects for the one being plotted.
    
    NOTE- The plots are not normalized wrt each other. This is intentional.
    
    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    saveplot : boolean
        Save the plot.
    species : int
        Index of species.

    Returns
    -------
    None.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Distribution function data
    f,vPar,vPerp=dist_func(filename,species=species)
    
    #Number of radial surface bins (=rdim)
    lrz=int(ds['lrz'][:])
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    #Array with the labels for each species
    speciesLabels=species_labels(filename)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    #Generate the savename of the plot
    #Get the name of the .nc file
    ncName=filename.split('/')[-1]
    #Remove the .nc part
    ncName=ncName[0:-3]
    savename=ncName+'_dist_func_species_'+speciesLabels[species]+'.pdf'
    
    #Initialize the plot
    fig,axs=plt.subplots(lrz,1,figsize=(21,lrz*9))
    
    #Go over each flux surface
    for fluxsurfplot in range(0,lrz):
    
        #Represent the r/a value in scientific notation
        ryaSciNot="{:.2e}".format(rya[fluxsurfplot])
        
        #Convert data to log
        logData=np.log10(f[fluxsurfplot])
        
        #Maximum of the distribution
        maxDist=np.max(logData)
        minDist=maxDist-15
        
        #Create the plot
        ax=axs[fluxsurfplot]
        pltobj=ax.contourf(vPar[fluxsurfplot],vPerp[fluxsurfplot],logData,levels=np.linspace(minDist,maxDist,30))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel(r'$v_{||}$ [m/s]')
        ax.set_xlim(-8e6,8e6)
        ax.set_xticks(np.linspace(-8e6,8e6,17))
        ax.set_ylabel(r'$v_{\perp}$ [m/s]')
        ax.set_ylim(0,8e6)
        ax.set_title('Distribution Function (r/a = '+ryaSciNot+')')
        ax.grid(True)
        cbar=fig.colorbar(pltobj,ax=ax)
        cbar.set_label(r'log$_{10}$(v$^{-3}$)')
        
    #Save the plot
    if saveplot==True:
        plt.savefig(plotDest+savename,bbox_inches='tight')
    plt.show()

#%% Testbed

#Location of CQL3D output
filename='WHAM_VNS_gen3_large_50keV_NBI_HFS_2p9Tres_2x2MWrf_5keV.nc'

#Get the netCDF4 data
ds=nc.Dataset(filename)

#Get the distribution function data
#distData,vPar,vPerp=dist_func(filename,makeplot=True,saveplot=True,fluxsurfplot=3,species=1)

#Get the ion densities
# ndwarmz,ndfz,ndtotz,solrz,solzz=ion_dens(filename,makeplot=True,saveplot=True,species=0)

#Get the pressure profiles for a given species
# pressparz_d,pressprpz_d,pressz_d,solrz,solzz=pressure(filename,makeplot=True,saveplot=True,species=1)

#Get the total pressure
# pressparz,pressprpz,pressz,solrz,solzz=total_pressure(filename,makeplot=True,saveplot=True)

#Get the plasma beta
# betaArr,solrz,solzz=beta(filename,makeplot=True,saveplot=True)

#Get the fusion neutron flux
# fusArr,zArr=axial_neutron_flux(filename,makeplot=True,saveplot=True)

#Get the fusion reaction rate
# fusrxrt,tArr=fusion_rx_rate(filename,makeplot=True,saveplot=True)

#Plot all distribution functions
# plot_dist_funcs(filename,saveplot=True,species=1)

#%% Beta Profile

# fig=plt.figure(figsize=(20,8))
# ax=fig.add_subplot(111)
# pltobj=ax.contourf(solzz,solrz,betaArr,levels=50)
# ax.set_xlabel('Z [m]')
# ax.set_ylabel('R [m]')
# ax.set_title('Plasma Beta')
# ax.grid()
# fig.colorbar(pltobj)
# plt.show()

#%% Fusion neutron flux

# plt.figure(figsize=(15,8))
# plt.plot(zArr,fusArr)
# plt.xlabel('Z [m]')
# plt.xlim(-1,1)
# plt.ylabel(r'Fusion Neutron Flux [W/m$^2$sr]')
# plt.ylim(0,50)
# plt.grid()
# plt.savefig(plotDest+'fusion_neutron_flux_withRF.svg')
# plt.show()

#%% Distribution function

# #Pick the flux surface
# fluxSurf=0

# #Convert data to log
# logData=np.log(distData[fluxSurf])

# # logData=distData[fluxSurf]

# #Create the plot
# plt.figure(figsize=(21,8))
# plt.contourf(vPar[fluxSurf],vPerp[fluxSurf],logData,levels=np.linspace(-70,50,50))
# plt.xlabel(r'$v_{||}$ [cm/s]')
# plt.xlim(-8e8,8e8)
# plt.xticks(np.linspace(-8e8,8e8,17))
# plt.ylabel(r'$v_{\perp}$ [cm/s]')
# plt.ylim(0,8e8)
# plt.title('Distribution Function (Flux Surface '+str(fluxSurf)+')')
# plt.colorbar()
# plt.grid()
# # plt.savefig(plotDest+'dist_func_withoutRF.svg')
# plt.show()

#%% Fusion Reaction Rate

# fig=plt.figure(figsize=(8,8))
# ax=fig.add_subplot(111)
# ax.plot(tArr,fusrxrt[0],label=r'D + T --> n + $^4$He')
# ax.plot(tArr,fusrxrt[1],label=r'D + $^3$He --> p + $^4$He')
# ax.plot(tArr,fusrxrt[2],label=r'D + D --> n + $^3$He')
# ax.plot(tArr,fusrxrt[3],label='D + D --> p + T')
# ax.set_xlabel('Time [s]')
# ax.set_ylabel(r'Fusion Reaction Rate [s$^{-1}$]')
# ax.grid()
# ax.set_title('Fusion Reaction Rate')
# ax.legend(bbox_to_anchor=(1,1))
# plt.savefig(plotDest+'fus_rx_rate_withoutRF.svg',bbox_inches='tight')
# plt.show()

#%% Fast ion density

# fig=plt.figure(figsize=(20,8))
# ax=fig.add_subplot(111)
# pltobj=ax.contourf(solzz,solrz,ndfz,levels=50)
# ax.set_xlabel('Z [m]')
# ax.set_ylabel('R [m]')
# ax.grid()
# fig.colorbar(pltobj)
# plt.show()