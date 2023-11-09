import glob
import os
import time
import re
import shutil

def clean_files(folder, max_num, prefix='', postfix=''):
    # all_files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(postfix)]
    all_files = list(filter(lambda x: os.path.isfile(x) and re.fullmatch(rf'{prefix}(\d+){postfix}', os.path.basename(x)), glob.glob(os.path.join(folder, f'{prefix}*{postfix}'))))
    all_files.sort(key=os.path.getmtime, reverse=True)
    # print('AAA: ', all_files)
    # for older_file in all_files[:max(len(all_files) - max_num, 0)]:
    #     os.remove(older_file)
    # all_files = list(filter(os.path.isfile, glob.glob(os.path.join(folder, f'{prefix}*{postfix}'))))
    # sorted_files = sorted(all_files, key=os.path.getmtime, reverse=True)
    # for older_file in sorted_files[max_num:]:
    for older_file in all_files[max_num:]:
        print(older_file)
        os.remove(older_file)

def main():
    tmp_dir = './tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    Nfiles = 10
    for idx in range(Nfiles):
        time.sleep(1)
        with open(os.path.join(tmp_dir, f'training-state-{idx:05d}.pt'), 'w') as f:
            f.write(f'{idx}')
    shutil.copy(os.path.join(tmp_dir, f'training-state-{Nfiles-1:05d}.pt'),
                os.path.join(tmp_dir, f'training-state-latest.pt'))
    
    clean_files(tmp_dir, 3, prefix='training-state-', postfix='.pt')
    
if __name__ == '__main__':
    main()