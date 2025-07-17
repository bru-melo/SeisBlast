import numpy as np
import pandas as pd
import re
import pygmt
from obspy.geodetics import degrees2kilometers
import os
#functions used in the relocate-blasts notebook
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def quarry_select(region, quarry_locations_df, blast_catalogue_df, radius, seisblast_folder):
    """
    Relocate blasts using GMT select based on quarry locations and a specified radius.
    Parameters:
    quarry_locations_df (pd.DataFrame): DataFrame containing quarry locations with columns 'lat' and 'lon'.
    blast_catalogue_df (pd.DataFrame): DataFrame containing blast events with columns 'LON(deg)', 'LAT(deg)', and 'EVENT-ID'.
    radius (str): Radius for selection in GMT format (e.g., '10k' for 10 kilometers).
    Returns:
    tuple: A tuple containing:
        - relocated_blasts (pd.DataFrame): DataFrame of relocated blasts with columns 'lon_ev', 'lat_ev', 'id-ev', 'lon-quarry', 'lat-quarry'.
        - remain_blasts (pd.DataFrame): DataFrame of remaining blasts that were not relocated.
        - repeated_blasts (pd.DataFrame): DataFrame of blasts that were assigned to more than one quarry location.
    """

    relocated_blasts = pd.DataFrame()
    repeated_blasts = pd.DataFrame()
    result = pd.DataFrame()
    
    #convert the blast catalogue dataframe to lon, lat (ASCII) to load into gmt select
    blast_catalogue_df.to_csv(f'{seisblast_folder}/Eire_blasts.gmt', 
            columns=['lon','lat','ID'], 
            sep="\t", 
            header=False, 
            index=False)

    for index, row in quarry_locations_df.iterrows():
        quarry_lat = quarry_locations_df.loc[index, "lat"]
        quarry_lon = quarry_locations_df.loc[index, "lon"]
        quarry_loc = str(quarry_lon) + "/" + str(quarry_lat)

        result = pygmt.select(
            data=f'{seisblast_folder}/Eire_blasts.gmt',
            output_type="pandas",
            dist2pt=quarry_loc + "+d" + radius,
            region=region,
        )

        if not result.empty:
            result.columns = ['lon', 'lat', 'ID']
            result["new-lon"] = quarry_lon
            result["new-lat"] = quarry_lat
            result["dist"] = haversine_km(
                    result["lat"], result["lon"], quarry_lat, quarry_lon
                )
            relocated_blasts = pd.concat([relocated_blasts, result], ignore_index=True)

    if not relocated_blasts.empty:
        relocated_blasts.columns = [
            "lon_ev",
            "lat_ev",
            "id-ev",
            "lon-quarry",
            "lat-quarry",
            "dist (km)"
        ]
    
        # Filter values that appear more than once
        value_counts = relocated_blasts['id-ev'].value_counts()
        repeated_values = value_counts[value_counts > 1]
        
        # Check if there are no repeated values
        if not repeated_values.any():
            print('    ~~~the selection radius '+str(radius)+' found ' +str(relocated_blasts.shape[0])+' matches and no double assigments.')
        else: 
            # If there are repeated values, print the number of non-unique assignments
            print('    ~~~the selection radius '+str(radius)+' found ' +str(relocated_blasts.shape[0])+' matches and '+str(repeated_values.iloc[0].max())+' non-unique assigments.')
            repeated_blasts = relocated_blasts[relocated_blasts['id-ev'].isin(repeated_values.index)]
            relocated_blasts = relocated_blasts[~relocated_blasts['id-ev'].isin(repeated_values.index)]
        
        # Build new event list with the remaining blasts
        remain_blasts = blast_catalogue_df[~blast_catalogue_df['ID'].isin(relocated_blasts["id-ev"])]
        print("    ~~~out of "+str(blast_catalogue_df.shape[0])+" initial events, "+str(remain_blasts.shape[0])+" quarry blast events remain")
    else:
        # If no matches were found
        print('    ~~~the selection radius '+str(radius)+' found no matches.')
        
    return relocated_blasts, remain_blasts, repeated_blasts

#functions used to read and order event data and information within their bulletins

##################################################################
#function to create a phase pick list from the event bulletin file
##################################################################

def list_picks(catalogue,bulletin_dir,event_id):
    # Accept bulletin_dir as a string or a list of directories
    if isinstance(bulletin_dir, str):
        bulletin_dirs = [bulletin_dir]
    else:
        bulletin_dirs = bulletin_dir

    bulletin_path = None
    for dir_path in bulletin_dirs:
        candidate_path = os.path.join(dir_path, str(event_id) + ".bulletin")
        if os.path.isfile(candidate_path):
            bulletin_path = candidate_path
            break

    if bulletin_path is None:
        raise FileNotFoundError(f"Bulletin file for event_id {event_id} not found in provided directories.")

    with open(bulletin_path) as bulletin:
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