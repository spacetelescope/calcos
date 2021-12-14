from __future__ import division         # confidence high

ta1image_range =  [8952, 8965]
ta1bright_range = [9419, 9425]
nuv_tv_dayrange = [264, 294]

fuv_osm1_dict = \
{7999: ("G130M", 1291, -2),
 8000: ("G130M", 1291, -1),
 8001: ("G130M", 1291, 0),
 8002: ("G130M", 1291, 1),
 7995: ("G130M", 1300, -2),
 7996: ("G130M", 1300, -1),
 7997: ("G130M", 1300, 0),
 7998: ("G130M", 1300, 1),
 7991: ("G130M", 1309, -2),
 7992: ("G130M", 1309, -1),
 7993: ("G130M", 1309, 0),
 7994: ("G130M", 1309, 1),
 7987: ("G130M", 1318, -2),
 7988: ("G130M", 1318, -1),
 7989: ("G130M", 1318, 0),
 7990: ("G130M", 1318, 1),
 7983: ("G130M", 1327, -2),
 7984: ("G130M", 1327, -1),
 7985: ("G130M", 1327, 0),
 7986: ("G130M", 1327, 1),
 11201: ("G160M", 1577, -2),
 11202: ("G160M", 1577, -1),
 11203: ("G160M", 1577, 0),
 11204: ("G160M", 1577, 1),
 11197: ("G160M", 1589, -2),
 11198: ("G160M", 1589, -1),
 11199: ("G160M", 1589, 0),
 11200: ("G160M", 1589, 1),
 11193: ("G160M", 1600, -2),
 11194: ("G160M", 1600, -1),
 11195: ("G160M", 1600, 0),
 11196: ("G160M", 1600, 1),
 11189: ("G160M", 1611, -2),
 11190: ("G160M", 1611, -1),
 11191: ("G160M", 1611, 0),
 11192: ("G160M", 1611, 1),
 11185: ("G160M", 1623, -2),
 11186: ("G160M", 1623, -1),
 11187: ("G160M", 1623, 0),
 11188: ("G160M", 1623, 1),
 1596: ("G140L", 1105, -2),
 1597: ("G140L", 1105, -1),
 1598: ("G140L", 1105, 0),
 1599: ("G140L", 1105, 1),
 1589: ("G140L", 1230, -2),
 1590: ("G140L", 1230, -1),
 1591: ("G140L", 1230, 0),
 1592: ("G140L", 1230, 1),
 1593: ("G140L", 1230, 0)
}

nuv_osm2_dict_early = \
{1304: ("G185M", 1817, 0),
 1283: ("G185M", 1850, -2),
 1284: ("G185M", 1850, -1),
 1285: ("G185M", 1850, 0),
 1286: ("G185M", 1850, 1),
 1267: ("G185M", 1882, 0),
 1245: ("G185M", 1921, 0),
 1226: ("G185M", 1953, 0),
 1208: ("G185M", 1986, 0),
 6421: ("G225M", 2217, 0),
 6399: ("G225M", 2250, -2),
 6400: ("G225M", 2250, -1),
 6401: ("G225M", 2250, 0),
 6402: ("G225M", 2250, 1),
 6390: ("G225M", 2268, 0),
 6381: ("G225M", 2283, 0),
 6356: ("G225M", 2325, 0),
 6337: ("G225M", 2357, 0),
 6344: ("G225M", 2357, 7),
 6318: ("G225M", 2390, 0),
 3949: ("G285M", 2637, 0),
 3939: ("G285M", 2657, 0),
 3929: ("G285M", 2676, 0),
 3912: ("G285M", 2709, 0),
 3838: ("G285M", 2850, -2),
 3839: ("G285M", 2850, -1),
 3840: ("G285M", 2850, 0),
 3841: ("G285M", 2850, 1),
 3793: ("G285M", 2952, 5),
 3788: ("G285M", 2952, 0),
 3774: ("G285M", 2979, 0),
 3754: ("G285M", 3018, 0),
 3751: ("G285M", 3035, 0),
 3734: ("G285M", 3057, 0),
 3728: ("G285M", 3074, -2),
 3729: ("G285M", 3074, -1),
 3730: ("G285M", 3074, 0),
 3731: ("G285M", 3074, 1),
11540: ("G230L", 2635, 0),
11519: ("G230L", 3000, -2),
11520: ("G230L", 3000, -1),
11521: ("G230L", 3000, 0),
11522: ("G230L", 3000, 1),
11502: ("G230L", 3360, 0)
}

nuv_osm2_dict_middle = \
{1324: ("G185M", 1786, 0),
 1306: ("G185M", 1817, -2),
 1307: ("G185M", 1817, -1),
 1308: ("G185M", 1817, 0),
 1309: ("G185M", 1817, 1),
 1298: ("G185M", 1835, 0),
 1287: ("G185M", 1850, -2),
 1288: ("G185M", 1850, -1),
 1289: ("G185M", 1850, 0),
 1290: ("G185M", 1850, 1),
 1282: ("G185M", 1864, 0),
 1272: ("G185M", 1882, 0),
 1267: ("G185M", 1890, 0),
 1261: ("G185M", 1900, -2),
 1262: ("G185M", 1900, -1),
 1263: ("G185M", 1900, 0),
 1264: ("G185M", 1900, 1),
 1255: ("G185M", 1913, 0),
 1249: ("G185M", 1921, -2),
 1250: ("G185M", 1921, -1),
 1251: ("G185M", 1921, 0),
 1252: ("G185M", 1921, 1),
 1240: ("G185M", 1941, 0),
 1231: ("G185M", 1953, -2),
 1232: ("G185M", 1953, -1),
 1233: ("G185M", 1953, 0),
 1234: ("G185M", 1953, 1),
 1223: ("G185M", 1971, 0),
 1214: ("G185M", 1986, 0),
 1150: ("G185M", 2010, -50),
 1175: ("G185M", 2010, -25),
 1190: ("G185M", 2010, -10),
 1200: ("G185M", 2010, 0),
 6432: ("G225M", 2186, -10),
 6442: ("G225M", 2186, 0),
 6424: ("G225M", 2217, 0),
 6412: ("G225M", 2233, -2),
 6413: ("G225M", 2233, -1),
 6414: ("G225M", 2233, 0),
 6415: ("G225M", 2233, 1),
 6404: ("G225M", 2250, 0),
 6393: ("G225M", 2268, 0),
 6384: ("G225M", 2283, 0),
 6372: ("G225M", 2306, 0),
 6358: ("G225M", 2325, -2),
 6359: ("G225M", 2325, -1),
 6360: ("G225M", 2325, 0),
 6361: ("G225M", 2325, 1),
 6352: ("G225M", 2339, 0),
 6340: ("G225M", 2357, 0),
 6329: ("G225M", 2373, -2),
 6330: ("G225M", 2373, -1),
 6331: ("G225M", 2373, 0),
 6332: ("G225M", 2373, 1),
 6321: ("G225M", 2390, 0),
 6309: ("G225M", 2410, 0),
 3959: ("G285M", 2617, 0),
 3947: ("G285M", 2637, -2),
 3948: ("G285M", 2637, -1),
 3949: ("G285M", 2637, 0),
 3950: ("G285M", 2637, 1),
 3940: ("G285M", 2657, 0),
 3931: ("G285M", 2676, 0),
 3919: ("G285M", 2695, -2),
 3920: ("G285M", 2695, -1),
 3921: ("G285M", 2695, 0),
 3922: ("G285M", 2695, 1),
 3914: ("G285M", 2709, 0),
 3908: ("G285M", 2719, -2),
 3909: ("G285M", 2719, -1),
 3910: ("G285M", 2719, 0),
 3911: ("G285M", 2719, 1),
 3900: ("G285M", 2739, 0),
 3834: ("G285M", 2850, -10),
 3842: ("G285M", 2850, -2),
 3843: ("G285M", 2850, -1),
 3844: ("G285M", 2850, 0),
 3845: ("G285M", 2850, 1),
 3854: ("G285M", 2850, 10),
 3791: ("G285M", 2952, -2),
 3792: ("G285M", 2952, -1),
 3793: ("G285M", 2952, 0),
 3794: ("G285M", 2952, 1),
 3780: ("G285M", 2979, 0),
 3771: ("G285M", 2996, 0),
 3760: ("G285M", 3018, 0),
 3739: ("G285M", 3057, 0),
 3720: ("G285M", 3094, 0),
11541: ("G230L", 2635, -2),
11542: ("G230L", 2635, -1),
11543: ("G230L", 2635, 0),
11544: ("G230L", 2635, 1),
11527: ("G230L", 2950, 0),
11522: ("G230L", 3000, -2),
11523: ("G230L", 3000, -1),
11524: ("G230L", 3000, 0),
11525: ("G230L", 3000, 1),
11503: ("G230L", 3360, -2),
11504: ("G230L", 3360, -1),
11505: ("G230L", 3360, 0),
11506: ("G230L", 3360, 1)
}

nuv_osm2_dict_late = \
{1319: ("G185M", 1786, 0),
 1303: ("G185M", 1817, 0),
 1293: ("G185M", 1835, 0),
 1284: ("G185M", 1850, 0),
 1277: ("G185M", 1864, 0),
 1267: ("G185M", 1882, 0),
 1262: ("G185M", 1890, 0),
 1258: ("G185M", 1900, 0),
 1250: ("G185M", 1913, 0),
 1246: ("G185M", 1921, 0),
 1235: ("G185M", 1941, 0),
 1228: ("G185M", 1953, 0),
 1218: ("G185M", 1971, 0),
 1209: ("G185M", 1986, 0),
 1195: ("G185M", 2010, 0),
 6437: ("G225M", 2186, 0),
 6419: ("G225M", 2217, 0),
 6409: ("G225M", 2233, 0),
 6399: ("G225M", 2250, 0),
 6388: ("G225M", 2268, 0),
 6379: ("G225M", 2283, 0),
 6367: ("G225M", 2306, 0),
 6355: ("G225M", 2325, 0),
 6347: ("G225M", 2339, 0),
 6335: ("G225M", 2357, 0),
 6326: ("G225M", 2373, 0),
 6316: ("G225M", 2390, 0),
 6304: ("G225M", 2410, 0),
 3839: ("G285M", 2850, 0),
11519: ("G230L", 3000, 0)
}