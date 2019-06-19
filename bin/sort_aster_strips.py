#!/usr/bin/env python
from __future__ import print_function
import os
import shutil
from datetime import datetime, timedelta
from glob import glob
import argparse
import numpy as np
from pybob.bob_tools import mkdir_p


def parse_aster_filename(fname):
    return datetime.strptime(fname[11:25], '%m%d%Y%H%M%S')


def sliding_window(a, size, step):
    for i in range(0, len(a)-1, step):
        yield(a[i:i+size])


def _argparser():
    parser = argparse.ArgumentParser(description="Sort ASTER scenes into folders based on whether \
    or not they form continuous strips.")
    parser.add_argument('--folder', action='store', type=str, help="Folder with raw, unsorted ASTER data.")
    parser.add_argument('--max_length', action='store', type=int, default=3,
                        help="Maximum number of scenes to put into a strip [3].")
    return parser


def main():
    parser = _argparser()
    args = parser.parse_args()
    
    if args.folder is not None:
        os.chdir(args.folder)
        
    print('Looking in folder {}'.format(os.getcwd()))

    flist = glob('*.zip.met')
    filenames = np.array([f.rsplit('.zip', 1)[0] for f in flist])
    filenames.sort()
    dates = [parse_aster_filename(f) for f in filenames]
    
    striplist = []
    # loop through the dates 
    for i, s in enumerate(dates):
        # get a list of all the scenes we're currently using
        current_striplist = [item for sublist in striplist for item in sublist]
        # if the current filename is already in the sorted list, move on.
        if filenames[i] in current_striplist:
            continue
        else:
            td_list = np.array([d - s for d in dates])
            # because we sorted the filelist, we don't have to consider timedeltas
            # less than zero (i.e., scenes within a single day are chronologically ordered)
            matched_inds = np.where(np.logical_and(td_list >= timedelta(0),
                                                   td_list < timedelta(0, 600)))[0]
            # if we only get one index back, it's the scene itself.
            if len(matched_inds) == 1:
                striplist.append(filenames[matched_inds])
                continue
            # now, check that we have continuity (if we have a difference of more than 12 seconds,
            # then the scenes aren't continuous even if they come from the same day)
            matched_diff = np.diff(np.array(td_list)[matched_inds])
            break_inds = np.where(matched_diff > timedelta(0, 12))[0]
            if len(break_inds) == 0:
                pass
            else:
                # we only need the first index, add 1 because of diff
                break_ind = break_inds[0] + 1
                matched_inds = matched_inds[0:break_ind]
            # here, we make sure that we only return strips that are at most max_length long.
            for strip in sliding_window(matched_inds, args.max_length, args.max_length-1):
                strip = list(strip)
                if len(matched_inds) > args.max_length and len(strip) == args.max_length-1:
                    strip.insert(0, strip[0]-1)                    
                striplist.append(filenames[strip])
    print('Found {} strips, out of {} individual scenes'.format(len(striplist), len(filenames)))
    # now that the individual scenes are sorted into "strips",
    # we can create "strip" and "single" folders
    print('Moving strips to individual folders.')
    #mkdir_p('strips')
    #mkdir_p('singles')
    mkdir_p('sorted')

    for s in striplist:
        mkdir_p(os.path.sep.join(['sorted', s[0][0:25]]))
        if len(s) == 1:
            # shutil.move(s[0] + '.zip', 'singles')
            # shutil.move(s[0] + '.zip.met', 'singles')
            shutil.move(s[0] + '.zip', os.path.sep.join(['sorted', s[0][0:25]]))
            shutil.move(s[0] + '.zip.met', os.path.sep.join(['sorted', s[0][0:25]]))
        else:
            #mkdir_p(os.path.join('strips', s[0][0:25]))
            for ss in s:
                # shutil.copy(ss + '.zip', os.path.join('strips', s[0][0:25]))
                # shutil.copy(ss + '.zip.met', os.path.join('strips', s[0][0:25]))
                shutil.copy(ss + '.zip', os.path.sep.join(['sorted', s[0][0:25]]))
                shutil.copy(ss + '.zip.met', os.path.sep.join(['sorted', s[0][0:25]]))
    # now, clean up the current folder.
    for f in glob('*.zip*'):
        os.remove(f)
    print('Fin.')


if __name__ == "__main__":
    main()