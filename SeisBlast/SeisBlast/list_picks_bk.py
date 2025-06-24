# seisblast/extract_picktimes.py

import re
import pandas as pd

#function to create a phase pick list from the event bulletin file
def list_picks(bulletin_dir,event_id):
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
    return event_picks_df
