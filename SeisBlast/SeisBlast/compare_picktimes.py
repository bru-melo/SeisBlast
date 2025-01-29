# SeisBlast/compare_picktimes.py

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import pygmt
import glob
from obspy import read, UTCDateTime
from obspy.geodetics import degrees2kilometers
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
#from obspy.imaging.spectrogram import spectrogram
from obspy.taup import TauPyModel
model = TauPyModel(model="ak135irelandcrust")
from scipy.signal import spectrogram, ShortTimeFFT
from scipy.signal.windows import gaussian


##################################################################
#function to create a phase pick list from the event bulletin file
##################################################################

def list_picks(catalogue,bulletin_dir,event_id):
    with open(bulletin_dir + str(event_id) + ".bulletin") as bulletin:
        phase_arrivals_data = []
        phase_arrivals_started = False
        lines = bulletin.readlines()

        for line in lines:
            line = line.strip()
            # Identify the phase arrivals section
            match = re.match(r'^(\d+)\s+Phase arrivals', line)
            if match:
                num_lines = int(match.group(1))
                phase_arrivals_started = True
                continue
            # Extract phase arrivals data
            if phase_arrivals_started:
                if line:  # Non-empty line
                    if not line.startswith('sta'):  # Skip the header line
                        parts = re.split(r'\s+', line)
                        if len(parts) == 10:  # Ensure we have the correct number of columns
                            phase_arrivals_data.append(parts)

                            #lines_read += 1
                #if lines_read >= num_lines:
                #    break

                elif line == '':  # Stop when an empty line is encountered
                    break

    # Create DataFrame for phase arrivals data
    phase_columns = ['sta', 'net', 'dist', 'azi', 'phase', 'time', 'res', 'wt', 'notsure','sta_repeat']
    event_picks_df = pd.DataFrame(phase_arrivals_data, columns=phase_columns)
    # Drop the redundant column
    event_picks_df.drop(columns=['sta_repeat'], inplace=True)
    # Convert times
    #read additional origin time to calculate start and end times
    event_info = catalogue.loc[catalogue.index[catalogue['EVENT-ID']== event_id].tolist()[0]]
    catalogue_ev = catalogue.loc[catalogue['EVENT-ID'] == event_id]
    #ev_date = UTCDateTime(catalogue_ev['DATE'].values[0] + catalogue_ev['TIME'].values[0])

    event_picks_df['time'] = pd.to_datetime((catalogue_ev['DATE'].values[0]+ ' ' + event_picks_df['time']),format='%Y-%m-%d %H:%M:%S.%f')
    event_picks_df['distkm'] = pd.to_numeric(event_picks_df['dist']).apply(degrees2kilometers)
    #reorganize table by station with P and S picks
    picks_df = event_picks_df.pivot_table(index=['sta','net','dist','distkm'],columns='phase', values=['time'],aggfunc='first').reset_index()

    return picks_df,event_info


##################################################################
# function to find the average pick time difference for a event
##################################################################

def diff_picks(catalogue,event_id,bulletin1,bulletin2):

    print('-- reading event id:',event_id,'--\n')

    ev_id_index = catalogue.index[catalogue['EVENT-ID'] == event_id].tolist()[0]
    ev_origin_time = pd.to_datetime((catalogue.loc[ev_id_index, 'DATE'] + ' ' + catalogue.loc[ev_id_index, 'TIME']), format='%Y-%m-%d %H:%M:%S.%f')

    #create pick list from bulletin files
    bul1_picks,info = list_picks(catalogue,bulletin1,event_id)
    bul2_picks,info = list_picks(catalogue,bulletin2,event_id)
    print('    > bulletin_1 has', bul1_picks.shape[0], 'times; bulletin_2 has', bul2_picks.shape[0])
    merge_picks = pd.merge(bul1_picks.sort_index(), bul2_picks.sort_index(), on=['sta','net'], how='inner', suffixes=['1','2'])
    merge_picks['time1_seconds'] = (merge_picks['time1']['P'] - ev_origin_time).dt.total_seconds()
    merge_picks['time2_seconds'] = (merge_picks['time2']['P'] - ev_origin_time).dt.total_seconds()
    #find time difference between the two pick lists
    merge_picks['time_diff_seconds'] = merge_picks['time2_seconds'] - merge_picks['time1_seconds']
    #filter all zero elements from dataframe
    zero_merge_picks = merge_picks[merge_picks['time_diff_seconds'] == 0].dropna(how='all').dropna(axis=1,how='all')

    if merge_picks.shape[0] == zero_merge_picks.shape[0] :
        print('    > all common station and phase picks are equal\n')
        return
    else :
        nonzero_merge_picks = merge_picks[merge_picks['time_diff_seconds'] != 0].dropna(how='all').dropna(axis=1,how='all')
        print('    > out of', merge_picks.shape[0], 'common station and phase picks.', nonzero_merge_picks.shape[0], 'pick times are different\n' )
        return nonzero_merge_picks

##################################################################
#function to read waveform from blast data
##################################################################

def get_wave(catalogue, event_id, bulletin_dir, i):
    
    #make list of picks per station
    picks_df,ev_date = list_picks(catalogue,bulletin_dir,event_id)

    #read the waveform
    ev_net = picks_df['net'][i]
    ev_station = picks_df['sta'][i]
    waveform = read('/mnt/REPO/MINISEED/' + str(ev_date.year) + '/'+ ev_net +'/'+ ev_station + '/HHZ.D/' + ev_net + '.' + ev_station + '..HHZ.D.' + str(ev_date.year) + '.' + str(ev_date.julday))
    
    return ev_date, picks_df, waveform

##################################################################
# simple view of the waveform with picks
##################################################################

def plot_wave(df,wf,origin,tmin,tmax,fmin,fmax,ind):

    #slice and filter the seismogram
    wf_filt = wf.copy()
    wf_filt.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    start = origin + tmin
    end = origin + tmax
    wf_slice = wf_filt.slice(start, end)
    relative_time = wf_slice[0].times(reftime=origin)

    #start plot figure
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(relative_time, wf_slice[0].data, "k-")
    ax.plot((UTCDateTime(df['time','P'][ind]) - origin),0,"|r",markersize=150,markeredgewidth=2, label='P')
    ax.plot((UTCDateTime(df['time','S'][ind]) - origin),0,"|g",markersize=150,markeredgewidth=2, label='S')
    ax.set_xlabel(f'Seconds relative to {origin}')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='lower right',markerscale=0.1)
    # Display the plot
    plt.show()


##################################################################
# subplots containing seismogram, spectrogram and wavelet transform
##################################################################

def plot_freq(df,info,tmin,tmax,fmin,fmax,sfmin,sfmax,ind,cfs=1,cfw=1,filterall=None):
#plot_freq: df dataframe containing the pick information
#           wf: trace data
#           info: event info from catalogue
#           tmin: start time from origin
#           tmax: end time from origin
#           fmin: high pass filter
#           fmax: low pass filter
#           sfmin: spectogram min frequency
#           sfmax: spectogram max frequency
#           ind: pick index from dataframe
#           clip_spec: max spectogram colorscale value
#           clip_wave: max wavelet colorscale value
#           filterall: if set use the same filter in the signal of the spectrograms

    #read the waveform
    ev_net = df['net'][ind]
    ev_station = df['sta'][ind]
    origin = UTCDateTime(info['DATE'] + info['TIME'])
    evdir = '/mnt/REPO/IRELAND/' + str(origin.year) + '/'+ ev_net +'/'+ ev_station + '/HHZ.D/'
    evfile = glob.glob(evdir + ev_net + '.' + ev_station + '.*.HHZ.D.' + str(origin.year) + '.' + str(origin.julday))
    wf = read(evfile[0])
    #wf = read('/mnt/REPO/MINISEED/' + str(origin.year) + '/'+ ev_net +'/'+ ev_station + '/HHZ.D/' + ev_net + '.' + ev_station + '..HHZ.D.' + str(origin.year) + '.' + str(origin.julday))
    
    # Set up the figure and subplots (one for the seismogram, one for the spectrogram, one for wavelet transform)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2, 2]})
    # Plot the seismogram in the top subplot (ax1)
    for trc in wf:
        #check for event time in multiple traces
        if trc.stats.starttime <= origin <= trc.stats.endtime:
            
            start = origin + tmin
            end = origin + tmax
            trace = trc.slice(start-5, end+5)
            tra_cp = trace.detrend('simple').copy()
            tra_cp.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
            tra_filt = tra_cp.slice(start, end)
            
            relative_times = tra_filt.times(reftime=origin)
            ax1.plot(relative_times, tra_filt.data, color='black')
            #plot P and S picked arrival
            #ax1.plot((UTCDateTime(df['time','P'][ind]) - origin),0,"|r",markersize=80,markeredgewidth=2, label='P')
            try:
                picktimeS = df.iloc[ind]['time','S']
                if pd.notna(picktimeS):
                    ax1.plot((UTCDateTime(df['time','S'][ind]) - origin),0,"|g",markersize=80,markeredgewidth=2, label='S')
                picktimeP = df.iloc[ind]['time','P']
                if pd.notna(picktimeP):
                    ax1.plot((UTCDateTime(df['time','P'][ind]) - origin),0,"|r",markersize=80,markeredgewidth=2, label='P')
            except:
                continue
            #plot P and S predicted arrivals
            arrivals = model.get_travel_times(source_depth_in_km=info['DEPTH(km)'],distance_in_degree=df.dist[ind],phase_list=["P","S"])
            ax1.plot(arrivals[0].time,0,"|b",markersize=60,markeredgewidth=1, label='Pmodel')
            ax1.plot(arrivals[1].time,0,"|c",markersize=60,markeredgewidth=1, label='Smodel')
            ax1.set_title(str(tra_filt.stats.station)+" "+str(tra_filt.stats.starttime)+" dist (km)"+str(df.distkm[ind]))
            ax1.set_ylabel("Amplitude")
            
            # Plot the spectrogram in the middle subplot (ax2)
            tra_unfilt = trace.slice(start, end).detrend('simple')
            in_trace = tra_filt if filterall=='Yes' else tra_unfilt
            #cax = spectrogram(in_trace.data, in_trace.stats.sampling_rate, axes=ax2, clip=[0.0,clip_spec], cmap='inferno')
            window_length=0.4 #controls the time resolution. smaller values improve the time (and less freq) and vice versa
            gauss_stdev=10 
            overlap=0.9 #higher is better, more time to compute
            sr = in_trace.stats.sampling_rate

            win = gaussian(int(window_length*sr), std=gauss_stdev, sym=True)  # symmetric Gaussian wind. 500=2.5s
            f, t, sxx = spectrogram(in_trace.data, fs=sr, window=win, noverlap=int(overlap * len(win)))
            clip_spec = 0.01*sxx.max()/cfs #adjusting of the color scale
            #ax2.imshow(sxx, origin='lower',aspect='auto', cmap='magma',vmin=0,vmax=clip_spec, extent=(0,len(in_trace.data)/sr,0,sr/2))
            ax2.imshow(sxx, origin='lower',aspect='auto', cmap='magma',vmin=0,vmax=clip_spec, extent=(tmin,tmax,0,sr/2))
            ax2.set_ylabel("Frequency [Hz]")
            ax2.set_ylim(sfmin, sfmax)
            
            #Plot the wavelet transform in the bottom subplot (ax3) (using the obspy option cwt)
            #CWT uses the morlet wavelet, with omega0 factor being the central frequency
            #higher w0 increases frequency resolution, lower w0 increases time resoltuion
            npts = in_trace.stats.npts
            dt = in_trace.stats.delta
            t = np.linspace(tmin, tmax, npts)
            #t = np.linspace(0, dt * npts, npts)
            w0 = 4
            scalogram = cwt(in_trace.data, dt, w0, sfmin, sfmax)
            #x, y = np.meshgrid(t,np.logspace(np.log10(sfmin), np.log10(sfmax), scalogram.shape[0]))
            x, y = np.meshgrid(t,np.linspace(sfmin,sfmax,scalogram.shape[0]))
            
            clip_spec = 0.5*np.abs(scalogram).max()/cfw
            ax3.pcolormesh(x, y, np.abs(scalogram), cmap='magma',vmin=[0.0,clip_spec])
            ax3.set_ylabel("Frequency [Hz]")
            #ax3.set_yscale('log')
            ax3.set_ylim(sfmin, sfmax)
            ax3.set_xlim(tmin, tmax)
            ax3.set_xlabel("Time from origin [s]")
            plt.show()
        else:
            continue
    return wf,in_trace

##################################################################
# plt map with event and stations
##################################################################

def eventmap(catalogue,df,ev_ind):
    
    region = [-10.5, -5, 51.5, 55.5]
    sta_columns = ['sta', 'name','lat', 'lon', 'elevation', 'net', 'end']
    stations = pd.read_csv("/mnt/store/senad/repos/for_bruna/QUARRY_BLASTS/ADMIN/Ireland_Britain_Stations", delimiter=' ', names=sta_columns)
    quarry = pd.read_csv("/mnt/store/senad/repos/for_bruna/QUARRY_BLASTS/ADMIN/IRELAND.quarry.coords", delimiter=' ')
    ev_st=stations.loc[stations['name'].isin(df['sta'])]
    ev_bls=catalogue.iloc[ev_ind]
    
    fig = pygmt.Figure()
    with fig.subplot(nrows=1, ncols=1, figsize=("6c","5c"), sharex="b", sharey="l", frame=["WSen","f50"]):
        # Subplot with original blast locations
        fig.basemap(region=region, projection="M?", panel=True)
        fig.coast(shorelines=True, water="skyblue", land="lightgray")
        
        fig.plot(
            x=quarry['LONGITUDE'],
            y=quarry['LATITUDE'],
            style="c0.02c",  # Circle with 0.3 cm size
            fill="blue",
            pen="blue",
            #label="Quarry"
        )
        # Plot the station locations
        fig.plot(
            x=ev_st['lon'],
            y=ev_st['lat'],
            style="t0.3c",  # Circle with 0.3 cm size
            fill="red",
            pen="black",
            #label="Stations"
        )

        fig.plot(
            x=ev_bls['LON(deg)'],
            y=ev_bls['LAT(deg)'],
            style="a0.3c", # Circle with 0.3 cm size
            fill="yellow",
            pen="black",
            #label="Blast Events"
        )
        fig.text(
            x=ev_st['lon'],
            y=ev_st['lat'],
            justify='BR',
            text=ev_st['name'],
            fill="white",
            pen="0.1p,red",
            font="3p,Helvetica"
        )
        # Show the figure
    fig.show()

##################################################################
# plt all stations for a event, scaled by distance
##################################################################

def seisdis(df,info,tmin,tmax,fmin,fmax,dfactor):
    
    origin = UTCDateTime(info['DATE'] + info['TIME'])
    plt.figure(figsize=(15,5))
    for i in range(len(df)):
        net = df.iloc[i]['net'].iloc[0]
        station = df.iloc[i]['sta'].iloc[0]
        evdir = '/mnt/REPO/IRELAND/' + str(origin.year) + '/'+ net +'/'+ station + '/HHZ.D/'
        evfile = glob.glob(evdir + net + '.' + station + '.*.HHZ.D.' + str(origin.year) + '.' + str(origin.julday))
        if evfile:
            stream = read(evfile[0])
        else:
            print("no trace found at " + evdir)
            continue
        for trace in stream:
            #check for event time in multiple traces
            if trace.stats.starttime <= origin <= trace.stats.endtime:
                start = origin + tmin
                end = origin + tmax
                tra_slice = trace.detrend('simple').slice(start-10, end+10)
                tra_slice.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
                tra_slice = tra_slice.slice(start,end)
                relative_times = tra_slice.times(reftime=origin)
                distance = eval(df.iloc[i]['dist'].iloc[0])
                #fig, ax = plt.subplots(figsize=(15, 5))
                plt.plot(relative_times,(tra_slice.data/np.max(np.abs(tra_slice.data))*dfactor)+distance,lw=0.6)
                #for when no pick time is available
                try:
                    picktime = df.iloc[i]['time','P']
                    if pd.notna(picktime):
                        plt.plot((UTCDateTime(df.iloc[i]['time','P']) - origin),distance,"|k",markersize=10,markeredgewidth=1)
                except:
                    continue
                plt.text(relative_times[-1]+0.05,distance,f"{station}",va='center')
                plt.xlabel('Time(s)')
                plt.ylabel('Amplitude')
                plt.xlim(tmin,tmax+3)
                break
            else:
                print("no trace with the event time")
            

def plot_psa(info,df,tmin,tmax,fmin,fmax):
#plot power spectra amplitude
    origin = UTCDateTime(info['DATE'] + info['TIME'])
    plt.figure(figsize=(8, 4))
    #all seismograms per event
    for i in range(len(df)):
        net = df.iloc[i]['net'].iloc[0]
        station = df.iloc[i]['sta'].iloc[0]
        repo_dir = '/mnt/REPO/MINISEED/' + str(origin.year) + '/'+ net +'/'+ station + '/HHZ.D/'
        #check if directory exists
        if not os.path.isdir(repo_dir):
            continue
        stream = read(repo_dir + net + '.' + station + '..HHZ.D.' + str(origin.year) + '.' + str(origin.julday))
        #check which trace contains the event
        for trace in stream:
            if trace.stats.starttime <= origin <= trace.stats.endtime:
                tra = trace.slice(origin+tmin, origin+tmax)
                    
                sampling_rate = tra.stats.sampling_rate
                n = tra.stats.npts  # number of points in the trace
                # Perform FFT
                fft_result = np.fft.fft(tra.data)
                #Only the first half is taken (up to Nyquist frequency) since the FFT result is symmetric for real-valued signals
                fft_amplitude = np.abs(fft_result)[:n // 2]  # take positive frequencies only
                # Calculate frequency axis
                #frequency bins corresponding to each FFT component, up to the Nyquist frequency (sampling rate / 2)
                frequencies = np.fft.fftfreq(n, d=1.0 / sampling_rate)[:n // 2]
                # Filter data to only include the specified frequency range
                freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
                filtered_frequencies = frequencies[freq_mask]
                filtered_amplitude = fft_amplitude[freq_mask]

                # Plot amplitude vs. frequency
                plt.plot(filtered_frequencies, filtered_amplitude,label=station)
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Amplitude")
                #plt.xlim(2,40)
                plt.autoscale(enable=True,axis='y')
                plt.grid(True)
                plt.legend()
                #plt.show()
                
            else:
                print('no trace')
#wavelet transform from mlpy
# from obspy import read, UTCDateTime
# import numpy as np
# origin = UTCDateTime(info['DATE'] + info['TIME'])
# tra = wf[0].slice(origin+10, origin + 40)
# trace = tra.copy().detrend()

# import math

# dj=0.05 #window spacing- maybe something like the overlap?
# N=trace.data.shape[0]
# dt=trace.stats.delta
# p=4 #wavelet function parameter (‘omega0’ for morlet, ‘m’ for paul and dog)
# wavelet="paul" #morlet, paul or dog

# scales = wvl.autoscales(N=N, dt=dt, dj=dj, wf=wavelet,p=p)
# spec = wvl.cwt(trace.data, dt=trace.stats.delta, scales=scales, wf=wavelet, p=p) 

# freq = (p + np.sqrt(2.0 + p**2)) / (4*np.pi * scales[1:])
# t = np.arange(trace.stats.npts) / trace.stats.sampling_rate
# fig, ax = plt.subplots(figsize=(10,5))
# test_frequencies = [freq[0], freq[-1], 40]
# #for f in test_frequencies:
# #    ax.axhline(f, color="white", linestyle="--", linewidth=5, label=f"f={f:.2f} Hz")
# plt.imshow(np.abs(spec), extent=(t[0], t[-1], freq[-1], freq[0]), vmin=0,vmax=200, cmap='magma', aspect='auto')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# #ax.legend(loc="upper right")
# plt.show()
# print("Scales range:", scales.min(), "to", scales.max())