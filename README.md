# SeisBlast
Python codes for sorting, visualising and processing quarry blast seismic data

Workflow

Checking picks and catalogues:

    0) compare-catalogues.ipynb
    -- compares whats similar between two event catalogues with similar event-ID (from SEISCOMP) to and compiles a catalogue of the repeating events.
    1) compare-picks.ipynb
    -- compare the picked traveltimes of the same event in the two catalogues. it loops over the diffent bulletin files to find where the picked times are the same and makes a list of the different time in seconds when a pick at a station at a event is different.
    2) visualize-picks.ipynb
    -- for each event fetches the picked travel times and plots it with the seismogram, spectogram and wavelet transform to access the quality of the picks.

Pre-processing and formating:
    
    3) relocate-blasts.ipynb
    -- for each quarry event location, we want to find the closest quarry and move the event location to that quarry.
    4) visualize-blasts.ipynb
    -- plots the station locations and event locations before and after relocation.
    5) inputsource-tomoatt.ipynb
    -- builds and formats the source receiver catalogue input file to run TOMOATT.
    6) select-traveltimes.ipynb
    -- finds the mean traveltime between each quarry-station pair.
    7) inputuniqsource-tomoatt.ipynb
    -- builds and formats the source receiver catalogue from the output from 6).