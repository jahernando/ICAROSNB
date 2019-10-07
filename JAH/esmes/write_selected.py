import os
import sys
import tables as tb
import numpy  as np
import pandas as pd
import warnings

from glob                                  import                     glob

from invisible_cities. core.     configure  import               configure
 
from invisible_cities. io.          dst_io  import                load_dst
from invisible_cities. io.         hits_io  import             hits_writer
from invisible_cities. io.         hits_io  import               load_hits
from invisible_cities. io.       mcinfo_io  import          mc_info_writer
from invisible_cities. io.run_and_event_io  import    run_and_event_writer
from invisible_cities. io. event_filter_io  import     event_filter_writer
from invisible_cities. io.          dst_io  import _store_pandas_as_tables


from invisible_cities. cities.  components  import   get_run_number
from invisible_cities. cities.  components  import   get_run_number
from invisible_cities. cities.  components  import   get_event_info
from invisible_cities. cities.  components  import get_mc_info_safe
from invisible_cities. cities.  esmeralda   import     track_writer
from invisible_cities. cities.  esmeralda   import   summary_writer

def read_and_select_events(infiles, event_list):
    for infile in infiles:
        with tb.open_file(infile, "r") as h5in:
            try:
                run_number  = get_run_number(h5in)
                event_info  = get_event_info(h5in)
                mc_info     = get_mc_info_safe(h5in, run_number)
            except (tb.exceptions.NoSuchNodeError, IndexError):
                continue
            event_intersection = list(set(event_list).intersection(event_info.read()['evt_number']))
            if event_intersection == []:
                continue
            try:
                hits         = load_hits(infile)
                tracks       = load_dst(infile, 'PAOLINA', 'Tracks')
                summary      = load_dst(infile, 'PAOLINA', 'Summary')
                hits_paolina = load_hits(infile, 'PAOLINA', 'Events')
            except (tb.exceptions.NoSuchNodeError, IndexError):
                continue
            for event_number in event_intersection:
                _, timestamp = event_info.read_where( '(evt_number == {})'.format(event_number))[0]
                yield dict(hits = hits[event_number], hits_paolina = hits_paolina[event_number],
                           tracks = tracks.loc[tracks.event==event_number],
                           summary = summary.loc[summary.event==event_number],
                           mc=mc_info, run_number=run_number,
                           event_number=event_number, timestamp=timestamp)

def loop_over_files_and_write(file_out, event_nums_file, folder_in, data = True, **kwargs):
    event_list = np.load(event_nums_file)
    filesin    = glob(folder_in +'/*.h5')[:100]
    with tb.open_file(file_out, 'w') as h5out:
        write_run_and_event = run_and_event_writer(h5out)
        write_mc_           = mc_info_writer(h5out) if (not data) else (lambda *_: None)
        write_hits_reco     = hits_writer (h5out)
        write_hits_paolina  = hits_writer(h5out, group_name = 'PAOLINA')
        write_tracks        = track_writer(h5out=h5out)
        write_summary       = summary_writer(h5out=h5out)
        
        events_generator = read_and_select_events(filesin, event_list)
        try:
            while True:
                selected = next(events_generator)
                write_run_and_event(selected['run_number'], selected['event_number'], selected['timestamp'])
                write_mc_(selected['mc'], selected['event_number'])
                write_hits_reco(selected['hits'])
                write_hits_paolina(selected['hits_paolina'])
                write_tracks(selected['tracks'])
                write_summary(selected['summary'])
        except StopIteration:
            return 0

if __name__ == "__main__":
    _, config_file = sys.argv
    conf = configure('dummy {}'.format(config_file).split())
    loop_over_files_and_write(**conf)
