import os


libfiles = ['libsig.py', 'libtsm.py']

folder_dict = {}

# Manually set the folders for each file
folder_dict['libsig.py'] = ['../gmc/']
folder_dict['libtsm.py'] = ['../gmc/']


# Print a confirm message
print(f'The current library hierachy is:')
# for each file, copy it in the corresponding folders
for lib in libfiles:
    print(f'Library {lib} to be copied into folders:')
    for folder in folder_dict[lib]:
        print(f'\t{folder}')

user_answer = input('Are you sure to proceed? (type y for yes)')
if user_answer == 'y':
    # for each file, copy it in the corresponding folders
    for lib in libfiles:
        for folder in folder_dict[lib]:
            path = folder + lib
            cmd = f'cp lib/{lib} {path}'
            os.system(cmd)
            print(f'{lib} copied into {folder}')
    print('Done.')
    print('DO NOT FORGET TO UPDATE WITH GIT')
else:
    print('Aborting.')
