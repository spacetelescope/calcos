# this module tests that the new col words are in all files

from datetime import datetime

from calcos import *

# list columns to search for
column_list = ['ERROR',
               'ERROR_LOWER',
               'VARIANCE_FLAT',
               'VARIANCE_COUNTS',
               'VARIANCE_BKG'
               ]

# cj edit: commented out this section since it is only for the initial testing

# get files to check
dir_prefix = '/grp/hst/cos/pipeline/pr_test/COSPIP-25/reduced_data/calcos_3.3.10.dev11+g8a6a0b7/'
dir_names = [dir_prefix + '11489',
             dir_prefix + '11667',
             dir_prefix + '13314',
             dir_prefix + '14673',
             dir_prefix + '14910',
             dir_prefix + '15778']
file_names = []
for d in dir_names:
    flist = glob.glob(d + '/*x1d*fits')
    file_names = file_names + flist

# cj edit: this section is for the regression test data
data_path = '/grp/hst/cos/Data/Testing/OPUS/regression_test/HSTDP-2020.5.0-15/'
file_names = glob.glob(data_path + '*x1d*fits')
file_names.sort()

# outfile = f test_column_existence_{datetime.now():%m%d%y}.output'  # for initial data
outfile = f'reg_test_column_existence_{datetime.now():%m%d%y}.output'  # for regression data
print(f"Saving Results to: {outfile}")
# with open(outfile,'w',1) as file_out:
#     pass_rate = 0
#     total_count = 0
#     for f in file_names:
#         with fits.open(f) as hdul:
#             data = hdul[1].data
#             hdr = hdul[0].header
#
#         for col in column_list:
#             total_count = total_count + 1
#             try:
#                 tarray = data[col]
#                 line = f"{f} ({hdr['DETECTOR']}/{hdr['OPT_ELEM']}/{hdr['OBSMODE']}) {col} : PASS \n"
#                 #line = f + ' ' + col + ': PASS' + '\n'
#                 file_out.write(line)
#                 pass_rate = pass_rate + 1
#             except KeyError:
#                 line = f"{f} ({hdr['DETECTOR']}/{hdr['OPT_ELEM']}/{hdr['OBSMODE']}) {col} : FAIL \n"
#                 #line = f + ' ' + col + ': FAIL' + '\n'
#                 file_out.write(line)
#     line = f'PASS RATE: {str((float(pass_rate)/float(total_count))*100.0)}% \nPASS NUMBER : {pass_rate}/{total_count}'
#     # line = 'PASS RATE: ' + str((float(pass_rate)/float(total_count))*100.0) + '\n'
#     file_out.write(line)
