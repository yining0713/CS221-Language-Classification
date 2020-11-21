#!/usr/bin/python3

import argparse
import csv
import os
import subprocess

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filelist", help="Name of the task")
    parser.add_argument("machine", help="Remote server alias")
    parser.add_argument("data_dir_remote", help="The top directory of the file to be copied")
    parser.add_argument("data_dir_local", help="The top directory of the file at destination")
    parser.add_argument("--ext", help="Extention the files if there are any", default=".mp3")
    args = parser.parse_args()
    return args


def main():
    args = read_args()
    with open(args.filelist, newline='') as f:
        csvreader = csv.reader(f, delimiter='\t')
        allfiles = [row[0] for idx, row in enumerate(csvreader)]
        alldir = set([os.path.dirname(x) for x in allfiles])
        print("Creating subdirectories...")
        for dir in alldir:
            absdir = os.path.join(args.data_dir_local, dir)
            retcode = subprocess.call(["mkdir", "-p", absdir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print("Copying audios...")
        for fileid in allfiles:
            file_abs_remote = os.path.join(args.data_dir_remote, fileid+args.ext) 
            file_abs_local = os.path.join(args.data_dir_local, fileid+args.ext)
            print(["rsync", "-aW", f"{args.machine}:{file_abs_remote}", file_abs_local])
            retcode = subprocess.call(["rsync", "-aW", f"{args.machine}:{file_abs_remote}", file_abs_local], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return


if __name__ == '__main__':
    main()
