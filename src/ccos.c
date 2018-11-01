/* This module contains the following functions:

binevents bins a list of (x,y) coordinates into a 2-D array.
bindq updates a 2-D array of data quality flags from DQI table info.
applydq assigns data quality flags from a DQI table into a column.
dq_or collapses (bitwise OR) a column of a 2-D data quality array to 1-D.
applyflat divides the epsilon values by a flat field.
range returns indices for a slice of time.
unbinaccum updates X & Y coordinates of pixel values in an image.
addrandom adds pseudo-random numbers on (-0.5,+0.5) to values in an array.
convolve1d convolves a 2-D image with a 1-D array.
extractband extracts a 2-D band (spectrum or background) from a 2-D image.
smoothbkg smoothes a 1-D array (background).
addlines creates a template spectrum based on a list of emission lines.
geocorrection applies the geometric (INL) distortion correction.
pha_check compares the pha with lower and upper limits.
clear_rows sets a temporary DQ array to 0 within curved boundaries.
interp1d does linear interpolation of one 1-D array onto another.
getstartstop gets indices of start and stop times of time intervals.
getbkgcounts gets the number of source and background counts within intervals.
smallerbursts screens for bursts (larger bursts screened separately).
getbadtime returns the sum of time intervals within which dq was bad.
xy_extract extracts a 1-D spectrum from an events list.
xy_collapse collapses events along the dispersion direction.
csum_3d bins a list of (x,y,pha) coordinates into a 3-D array.
csum_2d bins a list of (x,y) coordinates into a 2-D array.
bin2d bins a 2-D image to a smaller 2-D image (block sum).

2001 Nov 19
2001 Dec 7	In totalcounts, round float data to int.  In unbinaccum,
		add a check that the x and y arrays are long enough.
2002 Feb 1	Add function convolve1d.
2002 Feb 21	In binevents, applydq, and applyflat, allow x or y to be Int16.
2002 Mar 25	Add functions extractband and smoothbkg.
2002 Apr 12	Rename binmodule.c to ccosmodule.c.  Add function addlines.
2002 Apr 29	Add geocorrection.
2002 May 13	Add axis, mindopp, maxdopp to bindq.
2002 Sep 24	ndinfo->stride was being used incorrectly in get_f32, etc.;
		add the get2D and put2D functions.
2002 Nov 26	Convert from using NDInfo to using the NA macros.
2003 Jan 3	Fix typos.
2003 Feb 4	Rewrite applydq so it sets dq from DQI table, not from dq_array.
2003 Mar 19	Make changes in response to the code review.
2003 June 12	Include arrayobject.h instead of libnumarray.h, and call
		import_array() instead of import_libnumarray().
2004 Jan 9	In unbinImage, use an 'else' instead of an 'else if' to avoid
		a (spurious) compiler warning about counts possibly being used
		before being assigned a value.
		Use 'void' as the argument list for initcos.
		Remove nx and ny from extract2DBand.
		Add interp1d.
2004 Apr 8	Add doc strings.
2004 June 10	Include burst functions getbkgcounts and smallerbursts
		and getbadtime.
2004 July 7	Split getstartstop from getbkgcounts.
2004 Aug 3	Add xy_extract and xy_collapse.
2004 Oct 18	Remove unused variables.
2005 Mar 30	In getBkgCounts, fix bug regarding precedence of && and ||
		in the if statement for background regions.
2006 July 21	Change from numarray to numpy.
2007 Aug 29	Convert from numpy-compatible numarray to native numpy;
		remove y and dq from the calling sequence of getstartstop.
		In extract2DBand, interchange the loops over i and k, and
		take the computation of j (nearest integer to y) out of the
		inner loop; the reason is to ensure that each pixel in the
		cross-dispersion direction will be copied out exactly once,
		avoiding either skipped or duplicated pixels due to roundoff
		when computing nearest integer.  In smoothBackground, scr
		was allocated to be of length length+width * sizeof (float),
		but it should have been (length+width) * sizeof (float).
2008 Feb 8	getstartstop was not correctly handling the case that there
		were no events within a time interval.
2008 Mar 7	Add pixel_zero, epsilon (the weight column) and sdqflags as
		arguments to xy_extract.
2008 Oct 16	In getStartStopTimes, initialize istop to n_events instead of
		to zero, because the last element of istop won't otherwise
		be assigned if the time range isn't divisible by delta_t.
2008 Oct 23	Add functions csum_3d and csum_2d.
		Test for NULL return from PyArray_FROM_OTF.
2008 Nov 7	Fix a bug in getbadtime:  if the last event is flagged as bad,
		the last bad time interval was not included in the sum.
2009 Jan 22	Add argument x_offset to binevents, bindq, unbinaccum,
		extractband, and xy_extract.  Add clear_rows.
		Remove axis, mindopp, maxdopp from bindq, and replace
		dx, dy with ux, uy (upper limits, inclusive).
2009 Jan 23	In csum_2d, change the data type of the input x and y from
		int16 to float32 (because the xcorr & ycorr columns are now
		float32 for both FUV and NUV), and change both bin3DtoCsum
		and bin2DtoCsum to use nearest integer to convert the input
		x and y pixel coordinates to integer indices.
2009 Feb 23	Change ccos_range to convert time to float64, instead of
		accepting either float32 or float64.  Change timeRange to
		cast time to a float or double array (still allows either)
		when calling search or search_d, and change those functions
		to accept a float or double array rather than a PyArrayObject.
2009 May 6	Include binx and biny as arguments to csum_2d and csum_3d.
2009 May 13	Add functions dq_or and bin2d.
2010 Apr 2	In binarySearch, compare the first and last elements of array
		(rather than just the first and second elements) to check
		whether the array is increasing.
2010 Nov 22	Add function pha_check.
2011 Feb 17	Replace calls to malloc/free with PyMem_Malloc/PyMem_Free
		in functions findSmallerBursts and median_boxcar.
2011 June 29	Add an optional argument to smoothbkg, an array of flags;
		rewrite smoothBackground to use these flags.
2015 May 6      Add initialization code for Python 3
2015 May 20     Convert PyString function to PyUnicode
2017 Jan 30     Add bilinear_interpolation function for walk correction
*/

# include <Python.h>

# include <stdlib.h>
# include <string.h>
# include <math.h>
# include <limits.h>
# include <time.h>

# include <numpy/arrayobject.h>
# include <numpy/npy_3kcompat.h>

# define NPY_NO_DEPRECATED_API NPY_API_VERSION
# define SZ_ERRMESS 1024

/* This is the multiplier for the pseudo-random number generator.
   See Ahrens, J.H., Dieter, U., and Grube, A., "Pseudo-random Numbers.
   A New Proposal for the Choice of Multipliers," Computing 6, 1970,
   pp 121-138.  See also Dieter, U., "Pseudo-random Numbers:  The Exact
   Distribution of Pairs," Mathematics of Computation, Oct 1971,
   pp 855-883.
*/
# define RNG_MULTIPLIER 663608941

/* nearest integer function */
# define NINT(x)  ((int)(floor(x + 0.5)))

static char *DocString(void);

static PyObject *ccos_binevents(PyObject *, PyObject *);
static PyObject *ccos_bindq(PyObject *, PyObject *);
static PyObject *ccos_applydq(PyObject *, PyObject *);
static PyObject *ccos_dq_or(PyObject *, PyObject *);
static PyObject *ccos_applyflat(PyObject *, PyObject *);
static PyObject *ccos_range(PyObject *, PyObject *);
static PyObject *ccos_unbinaccum(PyObject *, PyObject *);
static PyObject *ccos_addrandom(PyObject *, PyObject *);
static PyObject *ccos_convolve1d(PyObject *, PyObject *);
static PyObject *ccos_extractband(PyObject *, PyObject *);
static PyObject *ccos_smoothbkg(PyObject *, PyObject *);
static PyObject *ccos_addlines(PyObject *, PyObject *);
static PyObject *ccos_geocorrection(PyObject *, PyObject *);
static PyObject *ccos_pha_check(PyObject *, PyObject *);
static PyObject *ccos_clear_rows(PyObject *, PyObject *);
static PyObject *ccos_interp1d(PyObject *, PyObject *);
static PyObject *ccos_getstartstop(PyObject *self, PyObject *args);
static PyObject *ccos_getbkgcounts(PyObject *, PyObject *);
static PyObject *ccos_smallerbursts(PyObject *, PyObject *);
static PyObject *ccos_getbadtime(PyObject *, PyObject *);
static PyObject *ccos_xy_extract(PyObject *, PyObject *);
static PyObject *ccos_xy_collapse(PyObject *, PyObject *);
static PyObject *ccos_csum_3d(PyObject *, PyObject *);
static PyObject *ccos_csum_2d(PyObject *, PyObject *);
static PyObject *ccos_walkcorrection(PyObject *, PyObject *);

static int binEventsToImage(PyArrayObject *, PyArrayObject *,
	PyArrayObject *, int, PyArrayObject *, short, PyArrayObject *);
static int binDQToImage(
	int [], int [], int [], int [],
	int [], int, PyArrayObject *, int);
static int applyDQToEvents(int [], int [],
		int [], int [], int [], int,
		PyArrayObject *, PyArrayObject *, short []);
static int bitwiseOrDQ(short [], short [], int, int);
static void applyFlatField(PyArrayObject *, PyArrayObject *,
	PyArrayObject *, PyArrayObject *, int, int);
static PyObject *timeRange(PyArrayObject *, double, double);
static int search(float [], int, float);
static int search_d(double [], int, double);
static int unbinImage(PyArrayObject *, int, float [], float[], int);
static int addRN(float [], int, int, int);
static int convolveWithDopp(PyArrayObject *, int, int, float [], int, int);
static int extract2DBand(PyArrayObject *,
	int, double, double, int, PyArrayObject *);
static int smoothBackground(int, int, float [], short []);
static int addEmissionLines(float [], double [], int,
		double, double [], float [], int);
static double findPixelNumber(double, double [], int);
static int binarySearch(double, double [], int);
static void addLSF(double, float, double, float [], int);
static int geoInterp2D(float [], float [], int,
	PyArrayObject *, PyArrayObject *, int, float, float, float, float);
static int phaCheck(int, short,
	float [], float [], short [], short [],
	PyArrayObject *, PyArrayObject *, int *, int *);
static int clearRows(PyArrayObject *,
	float [], float [], float [], float []);
static void bilinearInterp(float, float,
	PyArrayObject *, PyArrayObject *, int, int,
	float *, float *);
static int bilinearinterpolation(float x[], float y[],
        int n_events, PyArrayObject *image, float delta[]);
static int interp_check(PyArrayObject *, PyArrayObject *,
			 PyArrayObject *, PyArrayObject *);
static void interp1d(double [], double [], int, double [], double [], int);
static int getStartStopTimes(float [], int,
		int [], int [], int, double);
static int getBkgCounts(PyArrayObject *, short [], int,
		int [], int [], int [], int [], int,
		int, int, int, int,
		int, int, double);
static int findSmallerBursts(float [], short [], int,
		int[], int[], int[], int[], int, double,
		double, double, double,
		int, int,
		int, int, int, int);
static int median_boxcar(int [], int [], int, int, int);
static int compare_int(const void *, const void *);
static double getBadTime(float [], short [], int);
static int extrFromEvents(PyArrayObject *, PyArrayObject *, PyArrayObject *,
		int, double, double,
		PyArrayObject *, short, PyArrayObject *);
static int collapseFromEvents(PyArrayObject *, PyArrayObject *,
		short [], int,
		double, double [], int);
static void bin3DtoCsum(float [], int, int, int,
		int, int,
		float [], float [],
		float [], short [], int);
static void bin2DtoCsum(float [], int, int,
		int, int,
		float [], float [], float [], int);
static void bin2DArray(float [], int, int,
		float [], int, int);

/* This function returns the documentation string to be assigned to
   __doc__.
*/

static char *DocString(void) {

	return (
"This module contains the following functions:\n\n\
    binevents(x, y, array, x_offset,\n\
              <optional:  dq, sdqflags, epsilon>)\n\
    bindq(lx, ly, ux, uy, flag, dq_array, x_offset)\n\
    applydq(lx, ly, dx, dy, flag, x, y, dq)\n\
    dq_or(dq_2d, dq_1d)\n\
    applyflat(x, y, epsilon, flat,\n\
              <optional:  origin_x, origin_y>)\n\
    indices = range(time, t0, t1)\n\
    unbinaccum(image, x, y,\n\
               <optional:  x_offset>)\n\
    newseed = addrandom(x, seed, use_clock)\n\
    convolve1d(flat, dopp, axis)\n\
    extractband(indata, axis, slope, intercept, x_offset, outdata)\n\
    smoothbkg(data, width,\n\
              <optional:  flags>)\n\
    addlines(intensity, wavelength, reswidth, x1d_wl, dq, template)\n\
    geocorrection(x, y, x_image, y_image, interp_flag,\n\
                  <optional:  origin_x, origin_y, xbin, ybin>)\n\
    walkcorrection(fast, slow, refimage, delta)\n\
    counters = pha_check(x, y, pha, dq, im_low, im_high, pha_flag)\n\
    clear_rows(dq, y_lower, y_upper, x_left, x_right)\n\
    interp1d(x_a, y_a, x_b, y_b)\n\
    getstartstop(time, istart, istop, delta_t)\n\
    getbkgcounts(y, dq,\n\
                 istart, istop, bkg_counts, src_counts,\n\
                 bkg1_low, bkg1_high, bkg2_low, bkg2_high,\n\
                 src_low, src_high, bkgsf)\n\
    smallerbursts(time, dq,\n\
                  istart, istop, bkg_counts, src_counts,\n\
                  delta_t, smallest_burst, stdrej, source_frac,\n\
                  half_block, max_iter,\n\
                  large_burst, small_burst, dq_burst, verbose)\n\
    getbadtime(time, dq)\n\n\
    xy_extract(xi, eta, outdata, slope, intercept, x_offset,\n\
               <optional:  dq, sdqflags, epsilon>)\n\
    xy_collapse(xi, eta, dq, slope, xdisp)\n\
    csum_3d(array, x, y, epsilon, pha,\n\
            <optional:  binx, biny>)\n\
    csum_2d(array, x, y, epsilon,\n\
            <optional:  binx, biny>)\n\
    bin2d(array, binned_array)\n\
"
        /* string split because it is too long for windows compiler */
"\
x and y are arrays of pixel coordinates of the events (float32 or int16).\n\
x_offset is such that image pixel = detector coord + x_offset (int).\n\
epsilon is an array of weights for the events (float32).\n\
pha is an array of pulse height amplitudes (int16).\n\
dq is an array of data quality flags (0 is good; int16).\n\
array is the 2-D array modified in-place by binevents (float32).\n\
lx and ly are arrays of lower left corners of DQ regions (int32).\n\
dx and dy are arrays of DQ region widths (int32).\n\
flag is an array of data quality flags to assign to DQ regions (int16).\n\
dq_array is the 2-D array modified in-place by bindq (int16).\n\
mindopp and maxdopp are pixel offsets for Doppler shift (int).\n\
flat is a flat field (a 2-D array) (float32).\n\
time is the array of times of the events (float32 or float64).\n\
t0, t1 is a range of times within the time array (float).\n\
indices is a two-element tuple, the limits of the slice of time (int).\n\
image is a 2-D image array to be converted to a list of pixel coordinates\n\
    (int32, int16, uint16, or float32).\n\
ncounts is the sum of the pixel values in the image (int).\n\
seed is a 32-bit integer for starting the pseudo-random number generator.\n\
newseed is the value of seed after addrandom has been called.\n\
If use_clock is true, use the system clock to generate the seed.\n\
dopp is a 1-D array with which flat will be convolved (float32).\n\
axis (0 or 1) is the axis along which the convolution will be done (int).\n\
indata and outdata for extractband can be int16 or float32.\n\
pixel_zero is an offset to add to xi.\n\
For binevents, dq and epsilon are optional arguments.\n\
For bindq, axis, mindopp and maxdopp are optional arguments.\n");
}

/* calling sequence for binevents:

   binevents(x, y, array, x_offset, dq, sdqflags, epsilon)

    x, y       i: arrays of pixel coordinates of the events
                  (int16, or default is float32)
    array     io: the output 2-D array (float32)
    x_offset   i: the offset (it's zero or positive) to add to
                  the x pixel coordinate to get image pixel (int)

   optional arguments:
    dq         i: array of data quality flags (int16; 0 is good)
    sdqflags   i: bit mask for the "serious" dq flags (short)
    epsilon    i: array of weights for the events (float32)

   ccos_binevents calls binEventsToImage, which converts arrays of pixel
   coordinates to an image array.  The 2-D array ('array') will first be
   initialized to zero.  For each pair of elements (x[i],y[i]), the value
   in the nearest pixel to (x[i]+x_offset,y[i]) of array will be incremented.
   If epsilon is not null, the increment will be epsilon[i]; otherwise, the
   increment will be one.  If dq is not null, the pixel will be incremented
   only if dq[i] does not include a "serious" flag value (e.g. pulse height
   out of range or within a bad time interval).
*/

static PyObject *ccos_binevents(PyObject *self, PyObject *args) {

	PyObject *ox, *oy, *oarray, *odq, *oepsilon;
	PyArrayObject *x, *y, *array, *dq, *epsilon;
	int x_offset;
	short sdqflags;
	int status;

	odq = NULL;
	oepsilon = NULL;
	sdqflags = 32767;

	if (!PyArg_ParseTuple(args, "OOOi|OhO",
			&ox, &oy, &oarray, &x_offset,
			&odq, &sdqflags, &oepsilon)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	if (PyArray_TYPE(ox) == NPY_INT16) {
	    x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	if (PyArray_TYPE(oy) == NPY_INT16) {
	    y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	if (x == NULL || y == NULL)
	    return NULL;

	array = (PyArrayObject *)PyArray_FROM_OTF(oarray, NPY_FLOAT32,
			NPY_ARRAY_INOUT_ARRAY2);
	if (array == NULL)
	    return NULL;

	if (odq == NULL) {
	    dq = NULL;
	} else {
	    dq = (PyArrayObject *)PyArray_FROM_OTF(odq, NPY_INT16,
			NPY_IN_ARRAY);
	    if (dq == NULL)
		return NULL;
	}
	if (oepsilon == NULL) {
	    epsilon = NULL;
	} else {
	    epsilon = (PyArrayObject *)PyArray_FROM_OTF(oepsilon, NPY_FLOAT32,
			NPY_IN_ARRAY);
	    if (epsilon == NULL)
		return NULL;
	}

	status = binEventsToImage(x, y, array, x_offset,
			dq, sdqflags, epsilon);

	Py_DECREF(x);
	Py_DECREF(y);
	PyArray_ResolveWritebackIfCopy(array);
	Py_DECREF(array);
	Py_XDECREF(dq);
	Py_XDECREF(epsilon);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_binevents. */

static int binEventsToImage(PyArrayObject *x, PyArrayObject *y,
	PyArrayObject *array, int x_offset,
	PyArrayObject *dq, short sdqflags, PyArrayObject *epsilon) {

	float f_x_offset;	/* same as x_offset */
	int x_type, y_type;	/* data type codes */
	int n_events;		/* size of input arrays (number of events) */
	int nx, ny;		/* size of array */
	int k;			/* loop index for events */
	int i, j;		/* indices in 2-D array */
	/* individual values */
	float c_x;
	float c_y;
	short c_dq;
	float c_eps;

	/* The NINT macro should work the same way for both positive and
	   negative values.  To avoid any possibility of a discontinuity
	   at zero, however, for floating point data, f_x_offset will be
	   added to the x pixel coordinate (c_x) before rounding to an int.
	*/
	f_x_offset = (float)x_offset;

	x_type = x->descr->type_num;
	y_type = y->descr->type_num;

	n_events = PyArray_DIM(x, 0);
	nx = PyArray_DIM(array, 1);	/* shape (ny,nx) */
	ny = PyArray_DIM(array, 0);

	/* Initialize array to zero, because we're going to increment
	   a pixel value for each event in the list.
	*/
	for (i = 0;  i < nx;  i++)
	    for (j = 0;  j < ny;  j++)
		*(float *)PyArray_GETPTR2(array, j, i) = 0.;

	for (k = 0;  k < n_events;  k++) {

	    /* get the coordinates of the current event */
	    if (x_type == NPY_INT16) {
		i = *(short *)PyArray_GETPTR1(x, k);
		i += x_offset;
	    } else {
		c_x = *(float *)PyArray_GETPTR1(x, k);
		c_x += f_x_offset;
		i = NINT(c_x);		/* the more rapidly varying index */
	    }
	    if (y_type == NPY_INT16) {
		j = *(short *)PyArray_GETPTR1(y, k);
	    } else {
		c_y = *(float *)PyArray_GETPTR1(y, k);
		j = NINT(c_y);
	    }

	    if (dq == NULL)
		c_dq = 0;
	    else
		c_dq = *(short *)PyArray_GETPTR1(dq, k);

	    if ((c_dq & sdqflags) == 0) {

		if (epsilon == NULL)
		    c_eps = 1.;
		else
		    c_eps = *(float *)PyArray_GETPTR1(epsilon, k);

		/* truncate at borders of image */
		if (i < 0 || i >= nx || j < 0 || j >= ny)
		    continue;

		*(float *)PyArray_GETPTR2(array, j, i) += c_eps;
	    }
	}

	return 0;
}

/* calling sequence for bindq:

   bindq(lx, ly, ux, uy, flag, dq_array, x_offset)

    lx, ly     i: arrays of lower left corners of regions (int32)
    ux, uy     i: arrays of upper right corners of regions (int32)
    flag       i: array of data quality flags (int32)
    dq_array  io: 2-D array (int16); dq_array must have already been
                  initialized before calling bindq
    x_offset   i: the offset (it's zero or positive) to add to
                  x pixel coordinate to get the pixel in dq_array (int)

   The arrays lx, ly, ux, uy and flag are all of the same length.  These
   were taken from the data quality initialization table, but they may
   have been modified by adding wavecal offsets and Doppler shifts.
   Note that x_offset is added by this function.

   ccos_bindq calls binDQToImage, which updates in-place a 2-D DQ image
   array, flagging regions according to the regions specified in a data
   quality initialization table.  The input arrays are the lower left
   corner of a region, width of the region, and flag value.

   axis is the dispersion direction, which is relevant when extending the
   flagged region due to the Doppler shift during the exposure (specified
   by mindopp and maxdopp); if the value of axis is neither 0 nor 1, no
   Doppler smearing will be included.  mindopp and maxdopp are pixel offsets
   to be applied in the dispersion direction to each region to be flagged.

   dq_array will not be initialized by this function; values on input are
   assumed to be valid.
*/

static PyObject *ccos_bindq(PyObject *self, PyObject *args) {

	PyObject *olx, *oly, *oux, *ouy, *oflag, *odq_array;
	PyArrayObject *lx, *ly, *ux, *uy, *flag, *dq_array;
	int x_offset;
	int status;
	int nrows;

	if (!PyArg_ParseTuple(args, "OOOOOOi",
		&olx, &oly, &oux, &ouy, &oflag, &odq_array, &x_offset)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	lx = (PyArrayObject *)PyArray_FROM_OTF(olx, NPY_INT32, NPY_IN_ARRAY);
	ly = (PyArrayObject *)PyArray_FROM_OTF(oly, NPY_INT32, NPY_IN_ARRAY);
	ux = (PyArrayObject *)PyArray_FROM_OTF(oux, NPY_INT32, NPY_IN_ARRAY);
	uy = (PyArrayObject *)PyArray_FROM_OTF(ouy, NPY_INT32, NPY_IN_ARRAY);
	flag = (PyArrayObject *)PyArray_FROM_OTF(oflag, NPY_INT32,
			NPY_IN_ARRAY);
	if (lx == NULL || ly == NULL || ux == NULL || uy == NULL ||
		flag == NULL)
	    return NULL;
	dq_array = (PyArrayObject *)PyArray_FROM_OTF(odq_array, NPY_INT16,
			NPY_ARRAY_INOUT_ARRAY2);
	if (dq_array == NULL)
	    return NULL;

	nrows = PyArray_DIM(lx, 0);
	status = binDQToImage(
		(int *)PyArray_DATA(lx), (int *)PyArray_DATA(ly),
		(int *)PyArray_DATA(ux), (int *)PyArray_DATA(uy),
		(int *)PyArray_DATA(flag), nrows,
		dq_array, x_offset);

	Py_DECREF(lx);
	Py_DECREF(ly);
	Py_DECREF(ux);
	Py_DECREF(uy);
	Py_DECREF(flag);
	PyArray_ResolveWritebackIfCopy(dq_array);
	Py_DECREF(dq_array);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_bindq. */

static int binDQToImage(
	int lx[], int ly[], int ux[], int uy[],
	int flag[], int nrows, PyArrayObject *dq_array, int x_offset) {

	int nx, ny;		/* size of array */
	int k;			/* loop index for events */
	int i, j;		/* indices in 2-D array */
	/* individual values */
	int c_lx, c_ly;		/* lower left corner */
	int c_ux, c_uy;		/* upper right corner (from lx,ly and ux, uy) */
	int temp_flag;

	nx = PyArray_DIM(dq_array, 1);		/* shape (ny,nx) */
	ny = PyArray_DIM(dq_array, 0);

	for (k = 0;  k < nrows;  k++) {

	    c_lx = lx[k] + x_offset;
	    c_ly = ly[k];
	    c_ux = ux[k] + x_offset;
	    c_uy = uy[k];

	    /* ignore regions that are entirely out of bounds */
	    if (c_ux < 0 || c_lx >= nx || c_uy < 0 || c_ly >= ny)
		continue;

	    /* truncate at the array boundaries */
	    c_lx = (c_lx < 0) ? 0 : c_lx;
	    c_ly = (c_ly < 0) ? 0 : c_ly;
	    c_ux = (c_ux >= nx) ? nx-1 : c_ux;
	    c_uy = (c_uy >= ny) ? ny-1 : c_uy;

	    for (j = c_ly;  j <= c_uy;  j++) {
		for (i = c_lx;  i <= c_ux;  i++) {
		    temp_flag = *(short *)PyArray_GETPTR2(dq_array, j, i);
		    temp_flag |= flag[k];
		    *(short *)PyArray_GETPTR2(dq_array, j, i) =
				(short) temp_flag;
		}
	    }
	}

	return 0;
}

/* calling sequence for applydq:

   applydq(lx, ly, dx, dy, flag, x, y, dq)

    lx, ly     i: arrays of lower left corners of regions (int32)
    dx, dy     i: arrays of region widths (int32)
    flag       i: array of data quality flags (int32)
    x, y       i: arrays of pixel coordinates of the events, not Doppler
                  corrected (either int16 or float32)
    dq        io: array of data quality flags (int16)

   ccos_applydq calls applyDQToEvents, which updates in-place a 1-D dq
   array (column) for an events list, flagging regions according to the
   regions specified in a data quality initialization table (bpixtab).
   The input arrays from the reference table give the lower left corner
   of each region, width of the region, and flag value.  The x and y
   arrays are the rawx and rawy columns from an events table.  The dq
   array is the data quality column from that events table; this is
   the array that will be updated in-place.  Values of dq on input are
   assumed to be valid (i.e. this function will not initialize dq).
*/

static PyObject *ccos_applydq(PyObject *self, PyObject *args) {

	PyObject *olx, *oly, *odx, *ody, *oflag, *ox, *oy, *odq;
	PyArrayObject *lx, *ly, *dx, *dy, *flag, *x, *y, *dq;
	int status;
	int nrows;

	if (!PyArg_ParseTuple(args, "OOOOOOOO",
		&olx, &oly, &odx, &ody, &oflag, &ox, &oy, &odq)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	lx = (PyArrayObject *)PyArray_FROM_OTF(olx, NPY_INT32, NPY_IN_ARRAY);
	ly = (PyArrayObject *)PyArray_FROM_OTF(oly, NPY_INT32, NPY_IN_ARRAY);
	dx = (PyArrayObject *)PyArray_FROM_OTF(odx, NPY_INT32, NPY_IN_ARRAY);
	dy = (PyArrayObject *)PyArray_FROM_OTF(ody, NPY_INT32, NPY_IN_ARRAY);
	flag = (PyArrayObject *)PyArray_FROM_OTF(oflag, NPY_INT32,
			NPY_IN_ARRAY);
	if (lx == NULL || ly == NULL || dx == NULL || dy == NULL ||
		flag == NULL)
	    return NULL;
	if (PyArray_TYPE(ox) == NPY_INT16) {
	    x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	if (PyArray_TYPE(oy) == NPY_INT16) {
	    y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	dq = (PyArrayObject *)PyArray_FROM_OTF(odq, NPY_INT16,
			NPY_ARRAY_INOUT_ARRAY2);
	if (x == NULL || y == NULL || dq == NULL)
	    return NULL;

	nrows = PyArray_DIM(lx, 0);
	status = applyDQToEvents(
		(int *)PyArray_DATA(lx), (int *)PyArray_DATA(ly),
		(int *)PyArray_DATA(dx), (int *)PyArray_DATA(dy),
		(int *)PyArray_DATA(flag), nrows,
		x, y,
		(short *)PyArray_DATA(dq));

	Py_DECREF(lx);
	Py_DECREF(ly);
	Py_DECREF(dx);
	Py_DECREF(dy);
	Py_DECREF(flag);
	Py_DECREF(x);
	Py_DECREF(y);

	PyArray_ResolveWritebackIfCopy(dq);
	Py_DECREF(dq);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_applydq. */

static int applyDQToEvents(int lx[], int ly[],
		int dx[], int dy[], int flag[], int nrows,
		PyArrayObject *x, PyArrayObject *y,
		short dq[]) {

	int x_type, y_type;	/* data type codes for x and y */
	int i, j;		/* x & y rounded to int */
	int k;			/* loop index for events */
	int row;		/* loop index */
	int n_events;		/* number of rows in events table */
	/* individual values */
	int *c_ux, *c_uy;	/* array of upper right corners */
	float c_x;		/* rawx coordinate of an event */
	float c_y;		/* rawy coordinate of an event */
	short c_dq;		/* data quality column value */

	n_events = PyArray_DIM(x, 0);	/* rows in events table */
	x_type = x->descr->type_num;
	y_type = y->descr->type_num;

	c_ux = PyMem_Malloc(nrows * sizeof(int));
	c_uy = PyMem_Malloc(nrows * sizeof(int));
	if (c_ux == NULL || c_uy == NULL) {
	    PyErr_NoMemory();
	    return 1;
	}

	/* For each row in the data quality initialization (bpixtab)
	   reference table, get the location of the upper right corner.
	*/
	for (row = 0;  row < nrows;  row++) {
	    c_ux[row] = lx[row] + dx[row] - 1;
	    c_uy[row] = ly[row] + dy[row] - 1;
	}

	/* For each row in the events table, include the flag for each
	   region of the data quality initialization table.
	*/
	for (k = 0;  k < n_events;  k++) {

	    if (x_type == NPY_INT16) {
		i = *(short *)PyArray_GETPTR1(x, k);
	    } else {
		c_x = *(float *)PyArray_GETPTR1(x, k);
		i = NINT(c_x);		/* the more rapidly varying index */
	    }
	    if (y_type == NPY_INT16) {
		j = *(short *)PyArray_GETPTR1(y, k);
	    } else {
		c_y = *(float *)PyArray_GETPTR1(y, k);
		j = NINT(c_y);
	    }

	    /* Check each row of the bpixtab for overlap with (i,j). */
	    for (row = 0;  row < nrows;  row++) {
		if (i >= lx[row] && i <= c_ux[row] &&
		    j >= ly[row] && j <= c_uy[row]) {
		    c_dq = (short)flag[row];
		    dq[k] |= c_dq;
		}
	    }
	}

	PyMem_Free(c_ux);
	PyMem_Free(c_uy);

	return 0;
}

/* calling sequence for dq_or:

   dq_or(dq_2d, dq_1d)

    dq_2d      i: 2-D data quality array (int16)
    dq_1d     io: 1-D data quality array (int16)

   If the shape of dq_2d is (ny, nx), the shape of dq_1d should be (nx,).

   ccos_dq_or calls bitwiseOrDQ, which updates dq_1d in-place.  For each
   element i, dq_1d[i] will be the bitwise OR of dq_2d[:,i].
*/

static PyObject *ccos_dq_or(PyObject *self, PyObject *args) {

	PyObject *odq_2d, *odq_1d;
	PyArrayObject *dq_2d, *dq_1d;
	int nx, ny;
	int status;

	if (!PyArg_ParseTuple(args, "OO", &odq_2d, &odq_1d)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	dq_2d = (PyArrayObject *)PyArray_FROM_OTF(odq_2d, NPY_INT16,
			NPY_IN_ARRAY);
	dq_1d = (PyArrayObject *)PyArray_FROM_OTF(odq_1d, NPY_INT16,
			NPY_ARRAY_INOUT_ARRAY2);
	if (dq_2d == NULL || dq_1d == NULL)
	    return NULL;

	nx = PyArray_DIM(dq_2d, 1);	/* shape (ny,nx) */
	ny = PyArray_DIM(dq_2d, 0);
	if (nx != PyArray_DIM(dq_1d, 0)) {
	    PyErr_SetString(PyExc_RuntimeError,
		"dq_1d and dq_2d must have the same X axis length");
	    return NULL;
	}
	status = bitwiseOrDQ((short *)PyArray_DATA(dq_2d),
			     (short *)PyArray_DATA(dq_1d), nx, ny);

	Py_DECREF(dq_2d);
	PyArray_ResolveWritebackIfCopy(dq_1d);
	Py_DECREF(dq_1d);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_dq_or. */

static int bitwiseOrDQ(short dq_2d[], short dq_1d[], int nx, int ny) {

	int i, j;		/* array indices */
	short c_dq;		/* an individual value in dq_1d */

	for (i = 0;  i < nx;  i++) {
            dq_1d[i] = 0;
	}

	for (i = 0;  i < nx;  i++) {
	    c_dq = dq_1d[i];
	    for (j = 0;  j < ny;  j++) {
		c_dq |= dq_2d[i+nx*j];
	    }
	    dq_1d[i] = c_dq;
	}

	return 0;
}

/* calling sequence for applyflat:

   applyflat(x, y, epsilon, flat, origin_x, origin_y)

    x, y       i: arrays of pixel coordinates of the events
                  (either float32 or int16)
    epsilon   io: an array of efficiencies for the events (float32)
    flat       i: the 2-D flat field image array (float32)

   optional arguments:
    origin_x   i: offset in the more rapidly varying axis (int)
    origin_y   i: offset in the less rapidly varying axis (int)
            origin_x and origin_y are the offsets of the flat field array
            from the beginning of the detector, to allow the flat to be
	    a subarray.  origin_x & origin_y are the values of the keywords
	    ORIGIN_X and ORIGIN_Y respectively.  These are in units of
            pixels, and they are zero-indexed.  These are the negative of
            the IRAF keywords LTV1 & LTV2 respectively.

   ccos_applyflat calls applyFlatField, which applies a flat field image to
   the epsilon (weight) array.  It is assumed that epsilon has previously
   been initialized to one, although it may subsequently have been modified,
   e.g. by applying a deadtime correction.
*/

static PyObject *ccos_applyflat(PyObject *self, PyObject *args) {

	PyObject *ox, *oy, *oepsilon, *oflat;
	int origin_x=0, origin_y=0;
	PyArrayObject *x, *y, *epsilon, *flat;

	if (!PyArg_ParseTuple(args, "OOOO|ii",
			&ox, &oy, &oepsilon, &oflat, &origin_x, &origin_y)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	if (PyArray_TYPE(ox) == NPY_INT16) {
	    x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	if (PyArray_TYPE(oy) == NPY_INT16) {
	    y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_INT16,
			NPY_IN_ARRAY);
	} else {
	    y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_FLOAT32,
			NPY_IN_ARRAY);
	}
	epsilon = (PyArrayObject *)PyArray_FROM_OTF(oepsilon, NPY_FLOAT32,
			NPY_ARRAY_INOUT_ARRAY2);
	flat = (PyArrayObject *)PyArray_FROM_OTF(oflat, NPY_FLOAT32,
			NPY_IN_ARRAY);
	if (x == NULL || y == NULL || epsilon == NULL || flat == NULL)
	    return NULL;

	applyFlatField(x, y, epsilon, flat, origin_x, origin_y);

	Py_DECREF(x);
	Py_DECREF(y);
	PyArray_ResolveWritebackIfCopy(epsilon);
	Py_DECREF(epsilon);
	Py_DECREF(flat);

	Py_INCREF(Py_None);
	return Py_None;
}

/* This is called by ccos_applyflat. */

static void applyFlatField(PyArrayObject *x, PyArrayObject *y,
	PyArrayObject *epsilon, PyArrayObject *flat,
	int origin_x, int origin_y) {

	int x_type, y_type;	/* data type codes for x and y */
	int nx, ny;		/* size of flat */
	int k;			/* loop index for events */
	int n_events;		/* number of rows in events table */
	int i, j;		/* indices in 2-D array */
	/* individual values */
	float c_x, c_y, c_flat;

	x_type = x->descr->type_num;
	y_type = y->descr->type_num;

	n_events = PyArray_DIM(x, 0);	/* rows in events table */
	nx = PyArray_DIM(flat, 1);	/* shape (ny,nx) */
	ny = PyArray_DIM(flat, 0);

	/* For each event, find the location in the flat field, and
	   divide the current value of epsilon by the flat field value.
	*/
	for (k = 0;  k < n_events;  k++) {

	    /* get the coordinates of the current event */
	    if (x_type == NPY_INT16) {
		i = *(short *)PyArray_GETPTR1(x, k) - origin_x;
	    } else {
		c_x = *(float *)PyArray_GETPTR1(x, k);
		i = NINT(c_x) - origin_x;
	    }
	    if (y_type == NPY_INT16) {
		j = *(short *)PyArray_GETPTR1(y, k) - origin_y;
	    } else {
		c_y = *(float *)PyArray_GETPTR1(y, k);
		j = NINT(c_y) - origin_y;
	    }

	    /* ignore events that are outside the flat field image */
	    if (i < 0 || i >= nx || j < 0 || j >= ny)
		continue;

	    c_flat = *(float *)PyArray_GETPTR2(flat, j, i);
	    if (c_flat > 0.) {
		*(float *)PyArray_GETPTR1(epsilon, k) /= c_flat;
	    }
	}
}

/* calling sequence for range:

   indices = range(time, t0, t1)

    time       i: array of times of the events (read as float64)
    t0, t1     i: the times for which the indices are needed (double)

    indices    o: a two-element tuple of the indices in the time array
                  that bracket t0 and t1, to be used in the sense:
                      for i in range (indices[0], indices[1]):

   ccos_range calls timeRange, which returns (i0, i1) such that all values
   in time[i0:i1] are within the range from t0 to t1, and that would not
   be true for a smaller i0 or a larger i1.  time is assumed to be
   monotonically nondecreasing.
*/

static PyObject *ccos_range(PyObject *self, PyObject *args) {

	PyObject *otime;
	double t0, t1;
	PyArrayObject *time;
	PyObject *indices;

	if (!PyArg_ParseTuple(args, "Odd", &otime, &t0, &t1)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	time = (PyArrayObject *)PyArray_FROM_OTF(otime, NPY_FLOAT64,
		NPY_IN_ARRAY);
	if (time == NULL)
	    return NULL;

	indices = timeRange(time, t0, t1);

	Py_DECREF(time);

	return indices;
}

/* This is called by ccos_range. */

static PyObject *timeRange(PyArrayObject *time, double t0, double t1) {

	int i0, i1;
	int time_type;		/* data type code for time */
	double temp;		/* for swapping t0 & t1, if out of order */
	double tfirst, tlast;	/* first and last times in time */
	int n_events;		/* length of time array */
	PyObject *indices;

	/* Get the data type of the input time array. */
	time_type = time->descr->type_num;

	if (t1 < t0) {			/* swap, if out of order */
	    temp = t0;
	    t0 = t1;
	    t1 = temp;
	}

	n_events = PyArray_DIM(time, 0);

	/* get the times of the first and last events */
	if (time_type == NPY_FLOAT32) {
	    tfirst = *(float *)PyArray_GETPTR1(time, 0);
	    tlast = *(float *)PyArray_GETPTR1(time, n_events-1);
	} else {
	    tfirst = *(double *)PyArray_GETPTR1(time, 0);
	    tlast = *(double *)PyArray_GETPTR1(time, n_events-1);
	}

	if (t1 < tfirst || t0 > tlast) {
	    char errmess[SZ_ERRMESS+1];
	    sprintf(errmess,
		"(%.6g, %.6g) does not overlap the time array", t0, t1);
	    PyErr_SetString(PyExc_RuntimeError, errmess);
	    return NULL;
	}

	if (time_type == NPY_FLOAT32) {
	    i0 = search((float *)PyArray_DATA(time), n_events, (float)t0);
	    i1 = search((float *)PyArray_DATA(time), n_events, (float)t1);
	} else {
	    i0 = search_d((double *)PyArray_DATA(time), n_events, (float)t0);
	    i1 = search_d((double *)PyArray_DATA(time), n_events, (float)t1);
	}

	indices = Py_BuildValue("(i,i)", i0, i1);

	return indices;
}

/* This is a binary search function for t in time, but the routine isn't
   looking for an exact match.  We're searching for an index that can be
   used as the lower or upper index of a slice.

   If t falls between time[k] and time[k+1], then k+1 will be returned.
   If t exactly matches all time[k] values for k in the range k0 to k1
   inclusive, then k0 will be returned.
   if t is less than or equal to time[0], 0 will be returned.
   If t is greater than or equal to time[len(time)-1], len(time) will be
   returned (where len(time) is the length of the time array).  Note that
   the equality condition is inconsistent with the normal case where t is
   somewhere in the middle of the time array, but it makes the useage more
   intuitive.
*/

static int search(float time[], int n_events, float t) {

	int low, high;		/* current limits of search range */
	int mid;		/* middle of search range */
	float t_mid;		/* time at mid */

	low = 0;
	high = n_events - 1;

	if (t <= time[0])
	    return (0);

	if (t >= time[high])
	    return (high + 1);

	while (high - low > 1) {

	    mid = (low + high) / 2;
	    t_mid = time[mid];

	    if (t <= t_mid)
		high = mid;
	    else
		low = mid;
	}

	return (high);
}

/* This is a double precision version of search. */

static int search_d(double time[], int n_events, double t) {

	int low, high;
	int mid;
	double t_mid;

	low = 0;
	high = n_events - 1;

	if (t <= time[0])
	    return (0);

	if (t >= time[high])
	    return (high + 1);

	while (high - low > 1) {

	    mid = (low + high) / 2;
	    t_mid = time[mid];

	    if (t <= t_mid)
		high = mid;
	    else
		low = mid;
	}

	return (high);
}

/* calling sequence for unbinaccum:

   unbinaccum(image, x, y, x_offset)

    image      i: a 2-D array (int16, or default is float32)
    x, y      io: the arrays of pixel coordinates (float32)

   optional argument:
    x_offset   i: the offset (it's zero or positive) to subtract from the
                  image x pixel number to get the value to assign to x (int)

   ccos_unbinaccum calls unbinImage, which converts an image array ('image')
   to a pseudo time-tag list.  No time array will be created, just x and y
   pixel coordinates.  image is expected to contain integer values.  For
   each pixel of image that contains a positive value, that number of
   elements will be assigned in the x and y arrays, with their values
   being all the same, the coordinates of the current pixel in image.
*/

static PyObject *ccos_unbinaccum(PyObject *self, PyObject *args) {

	PyObject *oimage, *ox, *oy;
	PyArrayObject *image, *x, *y;
	int x_offset = 0;
	int status;
	int n_events;

	if (!PyArg_ParseTuple(args, "OOO|i", &oimage, &ox, &oy, &x_offset)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	if (PyArray_TYPE(oimage) == NPY_INT16) {
	    image = (PyArrayObject *)PyArray_FROM_OTF(oimage, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    image = (PyArrayObject *)PyArray_FROM_OTF(oimage, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_FLOAT32,
		NPY_ARRAY_INOUT_ARRAY2);
	y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_FLOAT32,
		NPY_ARRAY_INOUT_ARRAY2);
	if (image == NULL || x == NULL || y == NULL)
	    return NULL;

	n_events = PyArray_DIM(x, 0);
	if (PyArray_DIM(y, 0) < n_events)
	    n_events = PyArray_DIM(y, 0);
	status = unbinImage(image, x_offset,
		(float *)PyArray_DATA(x), (float *)PyArray_DATA(y), n_events);

	Py_DECREF(image);
	PyArray_ResolveWritebackIfCopy(x);
	PyArray_ResolveWritebackIfCopy(y);
	Py_DECREF(x);
	Py_DECREF(y);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_unbinaccum. */

static int unbinImage(PyArrayObject *image, int x_offset,
		float x[], float y[], int n_events) {

	float im_data_f32;		/* value before finding nearest int */
	int image_type;			/* data type code for image */
	int nx, ny;			/* size of image */
	int i, j, k;
	float ix, jy;			/* i and j converted to float */
	int counts;			/* value of image at a pixel */
	int n;				/* loop index over counts */

	image_type = image->descr->type_num;

	nx = PyArray_DIM(image, 1);		/* shape (ny,nx) */
	ny = PyArray_DIM(image, 0);

	/* Now extract counts into arrays. */
	k = 0;

	for (j = 0;  j < ny;  j++) {
	    for (i = 0;  i < nx;  i++) {

		if (image_type == NPY_INT16) {
		    counts = *(short *)PyArray_GETPTR2(image, j, i);
		} else {
		    im_data_f32 = *(float *)PyArray_GETPTR2(image, j, i);
		    counts = NINT(im_data_f32);
		}

		if (k+counts > n_events) {
		    PyErr_SetString(PyExc_RuntimeError,
				"x and y arrays are too short");
		    return 1;
		}

		/* these coordinates are zero indexed */
		ix = (float)i - x_offset;
		jy = (float)j;
		for (n = 0;  n < counts;  n++, k++) {
		    x[k] = ix;
		    y[k] = jy;
		}
	    }
	}

	return 0;
}

/* calling sequence for addrandom:

   newseed = addrandom(x, seed, use_clock)

    x         io: an array of pixel coordinates, either X or Y axis (float32)
    seed       i: a 32-bit integer for the pseudo-random number generator
    use_clock  i: true means use system clock to generate the seed (int)

    newseed    o: a 32-bit integer that could be used as the seed for
                  another call to addrandom

   ccos_addrandom calls addRn, which adds a pseudo-random number between
   -0.5 and +0.5 to each element of x.  If use_clock is true, the system
   clock will be used to create a seed for the generator (and the seed
   argument will be ignored); otherwise, seed will be used to start the
   generator.  The final value of the seed will be converted to a Python
   integer and returned as the function value.

   The actual endpoint of the range of the pseudo-random numbers may be
   equal to -0.5 or +0.5, depending on roundoff.
*/

static PyObject *ccos_addrandom(PyObject *self, PyObject *args) {

	PyObject *ox;
	PyArrayObject *x;
	int newseed;
	int seed;
	int use_clock;
	int n_events;

	if (!PyArg_ParseTuple(args, "Oii", &ox, &seed, &use_clock)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_FLOAT32,
		NPY_ARRAY_INOUT_ARRAY2);
	if (x == NULL)
	    return NULL;

	n_events = PyArray_DIM(x, 0);
	newseed = addRN((float *)PyArray_DATA(x), n_events, seed, use_clock);

	PyArray_ResolveWritebackIfCopy(x);
	Py_DECREF(x);

	return Py_BuildValue("i", newseed);
}

/* This is called by ccos_addrandom. */

static int addRN(float x[], int n_events, int seed, int use_clock) {

	double normalize;
	int k;

	/* Use the system clock to get a seed? */
	if (use_clock)
	    seed = time(NULL);

	/* Dividing by this normalization factor will make the pseudo-random
	   numbers cover the range from -0.5 to +0.5.
	*/
	normalize = 2. * (double)INT_MAX;

	for (k = 0;  k < n_events;  k++) {
	    seed *= RNG_MULTIPLIER;
	    x[k] += ((double)seed / normalize);
	}

	return seed;		/* return newseed */
}

/* calling sequence for convolve1d:

   convolve1d(flat, dopp, axis)

    flat      io: 2-D flat field image array (float32)
    dopp       i: 1-D array with which flat will be convolved (float32)
    axis       i: the axis (0 or 1) along which flat will be convolved (int);
                  axis 1 is the more rapidly varying axis

   The middle element of dopp corresponds to no shift.  For example,
   the following dopp would result in no change to flat:
        dopp = array ((0, 0, 0, 1, 0, 0, 0), dtype=float32)

   ccos_convolve1d calls convolveWithDopp, which convolves a flat field
   in-place with dopp.  The dopp array is 1-D, while the flat field is 2-D.
   The convolution will be done along just one axis, specified by axis.
*/

static PyObject *ccos_convolve1d(PyObject *self, PyObject *args) {

	PyObject *oflat, *odopp;
	PyArrayObject *flat, *dopp;
	int axis;
	int status;
	int nx, ny;
	int lendopp;

	if (!PyArg_ParseTuple(args, "OOi", &oflat, &odopp, &axis)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	flat = (PyArrayObject *)PyArray_FROM_OTF(oflat, NPY_FLOAT32,
		NPY_ARRAY_INOUT_ARRAY2);
	dopp = (PyArrayObject *)PyArray_FROM_OTF(odopp, NPY_FLOAT32,
		NPY_IN_ARRAY);
	if (flat == NULL || dopp == NULL)
	    return NULL;

	if (PyArray_NDIM(flat) > 2) {
	    PyErr_SetString(PyExc_RuntimeError, "flat must be only 2-D");
	    return NULL;
	}
	if (PyArray_NDIM(dopp) > 1) {
	    PyErr_SetString(PyExc_RuntimeError, "dopp must be only 1-D");
	    return NULL;
	}
	nx = PyArray_DIM(flat, 1);	/* shape (ny,nx) */
	ny = PyArray_DIM(flat, 0);
	lendopp = PyArray_DIM(dopp, 0);

	status = convolveWithDopp(flat, nx, ny,
		(float *)PyArray_DATA(dopp), lendopp, axis);

	PyArray_ResolveWritebackIfCopy(flat);

	Py_DECREF(flat);
	Py_DECREF(dopp);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_convolve1d. */

static int convolveWithDopp(PyArrayObject *flat, int nx, int ny,
		float dopp[], int lendopp, int axis) {

	int m;			/* middle of dopp */
	int i, j, k;
	float sum;
	float *c_flat;		/* a copy of one line or column of flat */

	if (axis == 1)
	    c_flat = PyMem_Malloc((nx + lendopp) * sizeof(float));
	else
	    c_flat = PyMem_Malloc((ny + lendopp) * sizeof(float));
	if (c_flat == NULL) {
	    PyErr_NoMemory();
	    return 1;
	}

	m = lendopp / 2;			/* truncate */

	if (axis == 1) {

	    /* Convolve along X, the more rapidly varying axis. */

	    /* It's actually just the m pixels at each end that need to be
		initialized; 1 is appropriate for convolving with a flat field.
	    */
	    for (i = 0;  i < nx+lendopp;  i++)
		c_flat[i] = 1.;

	    for (j = 0;  j < ny;  j++) {	/* for each image line */

		for (i = 0;  i < nx;  i++)
		    c_flat[m+i] = *(float *)PyArray_GETPTR2(flat, j, i);

		for (i = 0;  i < nx;  i++) {
		    sum = 0.;
		    for (k = 0;  k < lendopp;  k++)
			sum += (dopp[lendopp-1-k] * c_flat[i+k]);
		    *(float *)PyArray_GETPTR2(flat, j, i) = sum;
		}
	    }

	} else {

	    /* axis = 0:  convolve along Y, the less rapidly varying axis. */

	    for (j = 0;  j < ny+lendopp;  j++)
		c_flat[j] = 1.;

	    for (i = 0;  i < nx;  i++) {	/* for each image column */

		for (j = 0;  j < ny;  j++)
		    c_flat[m+i] = *(float *)PyArray_GETPTR2(flat, j, i);

		for (j = 0;  j < ny;  j++) {
		    sum = 0.;
		    for (k = 0;  k < lendopp;  k++)
			sum += (dopp[lendopp-1-k] * c_flat[j+k]);
		    *(float *)PyArray_GETPTR2(flat, j, i) = sum;
		}
	    }
	}

	PyMem_Free(c_flat);

	return 0;
}

/* calling sequence for extractband:

   extractband(indata, axis, slope, intercept, x_offset, outdata)

    indata     i: 2-D image array, from which a band will be extracted
                  (either float32 or int16)
    axis       i: the axis (0 or 1) along which the band will be extracted (int)
                  axis 1 is the more rapidly varying axis
    slope      i: the slope of the band (pixels per pixel, double)
    intercept  i: the center line of the band should cross this location
                  in the cross dispersion direction at pixel 'x_offset'
                  in the dispersion direction (pixel number, double)
    x_offset   i: x_offset is zero or positive; this is the offset of the
                  detector in the dispersion direction within indata (int)
    outdata   io: a 2-D array, into which the extracted data will be put
                  (either float32 or int16)

   While indata and outdata may be either float32 or int16, they both
   must be the same type.

   slope is the slope of the band with respect to the axis along which
   it will be extracted.  If axis=1 and slope=+0.1, for example, then
   the band is about six degrees counterclockwise from the horizontal
   axis.  If axis=0 and slope=+0.1, the band is about six degrees
   clockwise from the vertical axis.

   ccos_extractband calls extract2DBand, which copies out a 2-D array
   ('outdata') from a larger 2-D array ('indata').  outdata is an
   extracted spectrum or a background region.  The spectrum in indata
   may be oriented horizontally or vertically (specified by axis = 1 or 0
   respectively), but the data will be copied to outdata so that the
   dispersion direction in outdata will be oriented horizontally
   (the more rapidly varying axis).

   The location of the region in indata to be extracted is specified by
   slope and intercept (and axis).  The slope is in pixels per pixel.  The
   intercept is zero indexed, at the bottom (if axis=0) or left (if axis=1)
   edge; this is the location where the center line of the region to be
   extracted meets the edge.  No interpolation is done.  For each pixel in
   the dispersion direction, a short strip of pixels will be copied to
   outdata.  The location of that strip in the cross dispersion direction
   will be rounded to an integer, and pixel values will be copied directly
   to outdata.
*/

static PyObject *ccos_extractband(PyObject *self, PyObject *args) {

	PyObject *oindata, *ooutdata;
	int axis;
	double slope, intercept;
	int x_offset;
	PyArrayObject *indata, *outdata;
	int status;

	if (!PyArg_ParseTuple(args, "OiddiO",
			&oindata, &axis, &slope, &intercept, &x_offset,
			&ooutdata)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	if (axis < 0 || axis > 1) {
	    PyErr_SetString(PyExc_RuntimeError, "axis must be 0 or 1");
	    return NULL;
	}

	if (PyArray_TYPE(oindata) == NPY_INT16) {
	    indata = (PyArrayObject *)PyArray_FROM_OTF(oindata, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    indata = (PyArrayObject *)PyArray_FROM_OTF(oindata, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	if (indata == NULL)
	    return NULL;
	if (PyArray_TYPE(ooutdata) == NPY_INT16) {
	    outdata = (PyArrayObject *)PyArray_FROM_OTF(ooutdata, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    outdata = (PyArrayObject *)PyArray_FROM_OTF(ooutdata, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	if (outdata == NULL)
	    return NULL;

	status = extract2DBand(indata, axis, slope, intercept, x_offset,
			outdata);

	Py_DECREF(indata);
	Py_DECREF(outdata);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_extractband. */

static int extract2DBand(PyArrayObject *indata,
		int axis, double slope, double intercept,
		int x_offset,
		PyArrayObject *outdata) {

	int data_type;		/* data type code for indata and outdata */

	int length;		/* shape of outdata = (extr_height, length) */
	int width;		/* size of indata in cross-disp. direction */
	int extr_height;
	int half_height;	/* half of extr_height, fraction truncated */
	double y, y0;
	int i, j, k;		/* loop indices */
	int bounds_error;	/* true if band would be out of bounds */

	/* The intercept specified by the calling routine is the location
	   where the spectrum crosses the first column of the detector.
	   indata can be wider than the detector, in which case we change
	   the value of intercept to be where the spectrum crosses the first
	   column of indata.  x_offset is zero or positive.
	*/
	intercept -= slope * (double)x_offset;

	data_type = indata->descr->type_num;
	if (data_type != outdata->descr->type_num) {
	    PyErr_SetString(PyExc_RuntimeError,
			"indata and outdata must be of the same data type");
	    return 1;
	}

	/* this is expected to be odd, so truncate when dividing by two */
	extr_height = PyArray_DIM(outdata, 0);
	half_height = extr_height / 2;

	/* length is the axis length in the dispersion direction;
	   width is the axis length in the cross-dispersion direction.
	*/
	if (axis == 0) {
	    /* dispersion is in the vertical direction (NUV) */
	    length = PyArray_DIM(indata, 0);
	    width = PyArray_DIM(indata, 1);
	} else {
	    /* dispersion is in the horizontal direction (FUV) */
	    length = PyArray_DIM(indata, 1);
	    width = PyArray_DIM(indata, 0);
	}

	if (length != PyArray_DIM(outdata, 1)) {
	    PyErr_SetString(PyExc_RuntimeError,
		"second axis of outdata must agree with size of indata");
	    return 1;
	}

	/* Check for out of bounds. */
	bounds_error = 0;
	y0 = intercept - half_height;
	j = NINT(y0);
	if (j < 0)
	    bounds_error = 1;
	y0 = intercept - half_height + (length-1) * slope;
	j = NINT(y0);
	if (j < 0)
	    bounds_error = 1;

	y0 = intercept + half_height;
	j = NINT(y0);
	if (j >= width)
	    bounds_error = 1;
	y0 = intercept + half_height + (length-1) * slope;
	j = NINT(y0);
	if (j >= width)
	    bounds_error = 1;

	if (bounds_error) {
	    PyErr_SetString(PyExc_RuntimeError,
		"the band would extend beyond the boundary of indata");
	    return 1;
	}

	if (axis == 1) {			/* dispaxis = 1 */

	    for (i = 0;  i < length;  i++) {
		y = (intercept - half_height) + slope * i;
		j = NINT(y);
		for (k = 0;  k < extr_height;  k++, j++) {
		    /* output[k,i] = input[j,i] */
		    if (data_type == NPY_INT16) {
			*(short *)PyArray_GETPTR2(outdata, k, i) =
			*(short *)PyArray_GETPTR2(indata, j, i);
		    } else {
			*(float *)PyArray_GETPTR2(outdata, k, i) =
			*(float *)PyArray_GETPTR2(indata, j, i);
		    }
		}
	    }

	} else {				/* axis = 0, dispaxis = 2 */

	    for (i = 0;  i < length;  i++) {
		y = (intercept - half_height) + slope * i;
		j = NINT(y);
		for (k = 0;  k < extr_height;  k++, j++) {
		    /* output[k,i] = input[i,j] */
		    if (data_type == NPY_INT16) {
			*(short *)PyArray_GETPTR2(outdata, k, i) =
			*(short *)PyArray_GETPTR2(indata, i, j);
		    } else {
			*(float *)PyArray_GETPTR2(outdata, k, i) =
			*(float *)PyArray_GETPTR2(indata, i, j);
			/*                                ^^^^        */
			/* swapped, as compared with code for axis = 1 */
		    }
		}
	    }
	}

	return (0);
}

/* calling sequence for smoothbkg:

   smoothbkg(data, width, flags)

    data      io: a 1-D array to be smoothed in-place (float32)
    width      i: the width (pixels) of the boxcar smoothing function (int)

   optional argument:
    flags      i: a 1-D array of flags, 0 is good, 1 is bad (int16)

   ccos_smoothbkg calls smoothBackground, which boxcar smoothes the
   1-D array 'data' in-place, with a width of 'width' pixels.
*/

static PyObject *ccos_smoothbkg(PyObject *self, PyObject *args) {

	PyObject *odata, *oflags;
	PyArrayObject *data;
	PyArrayObject *flags;
	int width;
	int length;		/* length of data array */
	int status;

	oflags = NULL;

	if (!PyArg_ParseTuple(args, "Oi|O", &odata, &width, &oflags)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	data = (PyArrayObject *)PyArray_FROM_OTF(odata, NPY_FLOAT32,
		NPY_ARRAY_INOUT_ARRAY2);
	if (data == NULL)
	    return NULL;
	if (PyArray_NDIM(data) != 1) {
	    PyErr_SetString(PyExc_RuntimeError, "arrays must be 1-D");
	    return NULL;
	}

	length = PyArray_DIM(data, 0);

	if (oflags == NULL) {
	    short *dummy_flags;
	    int i;
	    dummy_flags = PyMem_Malloc(length * sizeof(short));
	    for (i = 0;  i < length;  i++)
		dummy_flags[i] = 0;
	    status = smoothBackground(length, width,
			(float *)PyArray_DATA(data), dummy_flags);
	    PyMem_Free(dummy_flags);
	} else {
	    flags = (PyArrayObject *)PyArray_FROM_OTF(oflags, NPY_INT16,
			NPY_IN_ARRAY);
	    if (flags == NULL) {
		Py_DECREF(data);
		return NULL;
	    }
	    if (PyArray_NDIM(flags) != 1) {
		PyErr_SetString(PyExc_RuntimeError, "flags must be 1-D");
		Py_DECREF(data);
		return NULL;
	    }
	    status = smoothBackground(length, width,
			(float *)PyArray_DATA(data),
			(short *)PyArray_DATA(flags));
	    Py_DECREF(flags);
	}

	PyArray_ResolveWritebackIfCopy(data);
	Py_DECREF(data);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_smoothbkg. */

static int smoothBackground(int length, int width,
		float data[], short flags[]) {

	double sum;		/* sum of elements not flagged as bad */
	double ngood;		/* number of good elements included in sum */
	float *scr_data;	/* temporary copy of data, extended at ends */
	short *scr_flags;	/* temporary copy of flags, extended */
	int offset;		/* width / 2, truncated */
	int ext_nelem;		/* length of extended arrays */
	int i, ilow, ihigh;
	int istart, iend;	/* first and last pixels not flagged as bad */

	ext_nelem = length + width;
	scr_data = PyMem_Malloc(ext_nelem * sizeof(float));
	scr_flags = PyMem_Malloc(ext_nelem * sizeof(short));
	if (scr_data == NULL || scr_flags == NULL) {
	    PyErr_NoMemory();
	    return 1;
	}
	memset(scr_data, 0, (ext_nelem) * sizeof(float));
	for (i = 0;  i < ext_nelem;  i++)
	    scr_flags[i] = 1;		/* initially all bad */

	offset = width / 2;

	/* copy to scratch */
	for (i = 0;  i < length;  i++) {
	    scr_data[i+offset] = data[i];
	    scr_flags[i+offset] = flags[i];
	}

	/* find first and last good pixels */
	istart = ext_nelem;		/* initial values outside array */
	iend = -1;
	for (i = 0;  i < ext_nelem;  i++) {
	    if (scr_flags[i] == 0) {
		istart = i;
		break;
	    }
	}
	for (i = ext_nelem-1;  i >= 0;  i--) {
	    if (scr_flags[i] == 0) {
		iend = i;
		break;
	    }
	}
	/* don't do any smoothing if there are no good pixels */
	if (istart >= ext_nelem || iend < 0)
	    return 0.;

	sum = 0.;
	ngood = 0.;
	for (i = 0;  i < width-1;  i++) {
	    if (scr_flags[i] == 0) {
		sum += scr_data[i];
		ngood++;
	    }
	}
	for (i = offset;  i < length+offset;  i++) {
	    ilow = i - offset - 1;
	    ihigh = ilow + width;
	    if (ihigh < ext_nelem) {
		if (scr_flags[ihigh] == 0) {
		    sum += scr_data[ihigh];
		    ngood++;
		}
	    }
	    if (ilow >= 0) {
		if (scr_flags[ilow] == 0) {
		    sum -= scr_data[ilow];
		    ngood--;
		}
	    }
	    if (i >= istart && i <= iend && ngood > 0.)
		data[i-offset] = sum / ngood;
	}

	PyMem_Free(scr_flags);
	PyMem_Free(scr_data);

	return 0;
}

/* calling sequence for addlines:

   addlines(intensity, wavelength, reswidth, x1d_wl, template)

    intensity   i: array of amplitudes of lines (float32)
    wavelength  i: array of wavelengths of lines (float64)
    reswidth    i: FWHM of Gaussian line shape (double)
    x1d_wl      i: wavelength array for template spectrum (read as float64)
    template   io: 1-D template spectrum (float32)

   ccos_addlines calls addEmissionLines, which constructs a spectrum of
   emission lines.  The wavelengths and intensities of the lines are
   given by the arrays intensity and wavelength; there is one element in
   each of those arrays for each emission line to be used in creating
   the spectrum.  The output spectrum (written to in-place) is
   template, and the wavelength at each pixel of template is
   given by x1d_wl.  The line shape is a Gaussian, boxcar smoothed
   by three pixels (to simulate a slit width of three pixels),
   and the full-width-half-maximum of the Gaussian is reswidth.
*/

static PyObject *ccos_addlines(PyObject *self, PyObject *args) {

	PyObject *ointensity, *owavelength, *ox1d_wl, *otemplate;
	double reswidth;
	PyArrayObject *intensity, *wavelength, *x1d_wl, *template;
	int status;
	int nrows;		/* length of intensity & wavelength arrays */
	int nelem;		/* length of x1d_wl and template arrays */

	if (!PyArg_ParseTuple(args, "OOdOO",
		&ointensity, &owavelength, &reswidth, &ox1d_wl, &otemplate)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	intensity = (PyArrayObject *)PyArray_FROM_OTF(ointensity,
			NPY_FLOAT32, NPY_IN_ARRAY);
	wavelength = (PyArrayObject *)PyArray_FROM_OTF(owavelength,
			NPY_FLOAT64, NPY_IN_ARRAY);
	x1d_wl = (PyArrayObject *)PyArray_FROM_OTF(ox1d_wl,
			NPY_FLOAT64, NPY_IN_ARRAY);
	template = (PyArrayObject *)PyArray_FROM_OTF(otemplate,
			NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);
	if (intensity == NULL || wavelength == NULL ||
		x1d_wl == NULL || template == NULL)
	    return NULL;

	nrows = PyArray_DIM(wavelength, 0);
	nelem = PyArray_DIM(x1d_wl, 0);
	if (nrows != PyArray_DIM(intensity, 0)) {
	    PyErr_SetString(PyExc_RuntimeError,
		"intensity and wavelength arrays are not the same length");
	    return NULL;
	}
	if (nelem != PyArray_DIM(template, 0)) {
	    PyErr_SetString(PyExc_RuntimeError,
		"x1d_wl and template arrays are not the same length");
	    return NULL;
	}

	status = addEmissionLines(
		(float *)PyArray_DATA(intensity),
		(double *)PyArray_DATA(wavelength), nrows,
		reswidth,
		(double *)PyArray_DATA(x1d_wl),
		(float *)PyArray_DATA(template), nelem);

	Py_DECREF(intensity);
	Py_DECREF(wavelength);
	Py_DECREF(x1d_wl);
	PyArray_ResolveWritebackIfCopy(template);
	Py_DECREF(template);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_addlines. */

static int addEmissionLines(
		float intensity[], double wavelength[], int nrows,
		double reswidth,
		double x1d_wl[], float template[], int nelem) {

	double x;		/* pixel number */
	double minwl, maxwl;	/* min and max wavelengths in template */
	double wl;		/* for swapping minwl and maxwl */
	int i;

	memset(template, 0, nelem * sizeof(float));

	minwl = x1d_wl[nelem-1];
	maxwl = x1d_wl[0];
	if (minwl > maxwl) {
	    wl = minwl;
	    minwl = maxwl;
	    maxwl = wl;
	}

	for (i = 0;  i < nrows;  i++) {
	    if (wavelength[i] <= minwl || wavelength[i] >= maxwl)
		continue;
	    if (intensity[i] <= 0.)
		continue;
	    x = findPixelNumber(wavelength[i], x1d_wl, nelem);
	    addLSF(reswidth, intensity[i], x, template, nelem);
	}

	return (0);
}

static double findPixelNumber(double wl, double x1d_wl[], int nelem) {

	double x;		/* pixel coordinate */
	int i;

	i = binarySearch(wl, x1d_wl, nelem);
	if (i == -1 || i == nelem)
	    x = (double)i;		/* out of range */
	else
	    x = i + (wl - x1d_wl[i]) / (x1d_wl[i+1] - x1d_wl[i]);

	return (x);
}

/* This function does a binary search and returns the index i such that
   wl is between array[i] and array[i+1].  Normally, i is restricted to
   the range from 0 to n-1 inclusive.

   Out of range conditions:

   i = -1 means that wl is smaller than array[0] (or larger, if array is
   decreasing); i = nelem means that wl is greater than array[nelem-1]
   (or smaller, if array is decreasing).  Note that the function value i
   will never be equal to nelem-1.
*/

static int binarySearch(double wl, double array[], int nelem) {

	int low, high;		/* range of elements to consider */
	int k;			/* middle element between low and high */
        int increasing;         /* true if array is increasing */

	if (nelem < 2)
	    return 0;

	if (array[0] < array[nelem-1]) {	/* array is increasing */

	    increasing = 1;

	    /* check for out of range */
	    if (wl < array[0])
		return -1;
	    else if (wl > array[nelem-1])
		return nelem;

	} else {				/* array is decreasing */

	    increasing = 0;

	    if (wl > array[0])
		return -1;
	    else if (wl < array[nelem-1])
		return nelem;
	}

	low = 0;
	high = nelem - 1;

	if (increasing) {

	    while (high - low > 1) {
		k = (low + high) / 2;
		if (wl > array[k])
		    low = k;
		else
		    high = k;
	    }

	} else {

	    while (high - low > 1) {
		k = (low + high) / 2;
		if (wl > array[k])
		    high = k;
		else
		    low = k;
	    }
	}

	return (low);
}

/* MAX_LEN_TEMP and MIN_LEN_TEMP are only used by addLSF. */
# define MAX_LEN_TEMP 81
# define MIN_LEN_TEMP 21

/* This function adds an emission line with amplitude ampl, centered at
   pixel number x, to template.  The line shape is a Gaussian, boxcar
   smoothed by three pixels; the FWHM is reswidth, and the line is
   truncated after about 2.5 * reswidth.
*/

static void addLSF(double reswidth, float ampl, double x,
		float template[], int nelem) {

/* arguments:
double reswidth      i: FWHM of a Gaussian (pixels)
double x             i: pixel location of the emission line
float template[]    io: template spectrum; modified in-place
int nelem            i: size of template
*/

	double sigma;
	double dx, x2;
	float temp[MAX_LEN_TEMP];	/* scratch for Gaussian function */
	int len_temp;		/* length of temp that we use */
	int mid;		/* middle pixel of temp */
	int ix;			/* x truncated to an integer */
	int i;			/* index in temp */
	int j;			/* index in template */

	if (x < 0. || x >= nelem)
	    return;

	len_temp = 2 * (int)(2.5 * reswidth) + 1;
	len_temp = (len_temp <= MAX_LEN_TEMP ? len_temp : MAX_LEN_TEMP);
	len_temp = (len_temp >= MIN_LEN_TEMP ? len_temp : MIN_LEN_TEMP);

	/* After filling temp with the Gaussian function, we can add ...
	   ...
	   temp[mid-1] to template[ix-1],
	   temp[mid]   to template[ix],
	   temp[mid+1] to template[ix+1],
	   ...
	*/
	mid = len_temp / 2;
	ix = (int) floor(x);
	dx = x - (double)ix;

	/* fwhm / 2 = sigma * sqrt(2 * ln(2)) */
	sigma = reswidth / 2.35482;

	for (i = 0;  i < len_temp;  i++) {
	    x2 = (double)(i - mid) - dx;
	    temp[i] = ampl * exp(-(x2*x2) / (2.*sigma*sigma));
	}

	/* Add to template, assuming a slit width of three pixels. */
	for (i = 1;  i < len_temp-1;  i++) {
	    j = i - mid + ix;
	    if (j < 0 || j >= nelem)
		continue;
	    template[j] += (temp[i-1] + temp[i] + temp[i+1]) / 3.;
	}
}

/* calling sequence for geocorrection:

   geocorrection(x, y, x_image, y_image, interp_flag, \
		origin_x, origin_y, xbin, ybin)

    x, y       io: arrays of pixel coordinates of the events (float32)
    x_image     i: the 2-D image array of dx values (float32)
    y_image     i: the 2-D image array of dy values (float32)
    interp_flag i: indicates the type of interpolation (int):
                   0 --> use nearest neighbor, 1 --> use bilinear interpolation

   optional arguments:
    origin_x    i: offset in the more rapidly varying axis (int)
    origin_y    i: offset in the less rapidly varying axis (int)
    xbin        i: bin factor in the more rapidly varying axis (int)
    ybin        i: bin factor in the less rapidly varying axis (int)
            origin_x and origin_y are the offsets of x_image and y_image
            from the beginning of the detector, to allow them to be
            subarrays.  origin_x & origin_y are the values of the keywords
            ORIGIN_X and ORIGIN_Y respectively.  These are in units of
            unbinned pixels, and they are zero-indexed.
            xbin and ybin are the bin factors of x_image and y_image
            in the X and Y axes respectively, with default values of 1.
            These are the values of the keywords XBIN and YBIN.

   ccos_geocorrection calls geoInterp2D, which applies the geometric (INL)
   correction to the x and y arrays (in-place).  The corrections to x and y
   are given by the values in x_image and y_image respectively, which have
   offsets of origin_x and origin_y from the origin (so the images do not
   have to be full size).  If interp_flag is true, bilinear interpolation
   within x_image and y_image will be used to get the corrections to x and
   y; otherwise, the nearest pixel in x_image and y_image will be used.
*/

static PyObject *ccos_geocorrection(PyObject *self, PyObject *args) {

	PyObject *ox, *oy, *ox_image, *oy_image;
	int interp_flag;
	int origin_x, origin_y;
	int xbin, ybin;
	PyArrayObject *x, *y, *x_image, *y_image;
	int status;
	int n_events;		/* number of rows in events table */

	origin_x = 0;
	origin_y = 0;
	xbin = 1;
	ybin = 1;

	if (!PyArg_ParseTuple(args, "OOOOi|iiii",
			&ox, &oy, &ox_image, &oy_image, &interp_flag,
			&origin_x, &origin_y, &xbin, &ybin)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_FLOAT32,
			NPY_ARRAY_INOUT_ARRAY2);
	y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_FLOAT32,
			NPY_ARRAY_INOUT_ARRAY2);
	x_image = (PyArrayObject *)PyArray_FROM_OTF(ox_image, NPY_FLOAT32,
			NPY_IN_ARRAY);
	y_image = (PyArrayObject *)PyArray_FROM_OTF(oy_image, NPY_FLOAT32,
			NPY_IN_ARRAY);
	if (x == NULL || y == NULL || x_image == NULL || y_image == NULL)
	    return NULL;

	n_events = PyArray_DIM(x, 0);	/* rows in events table */
	status = geoInterp2D(
		(float *)PyArray_DATA(x), (float *)PyArray_DATA(y), n_events,
		x_image, y_image, interp_flag,
		(float)origin_x, (float)origin_y, (float)xbin, (float)ybin);

	PyArray_ResolveWritebackIfCopy(x);
	PyArray_ResolveWritebackIfCopy(y);

	Py_DECREF(x);
	Py_DECREF(y);
	Py_DECREF(x_image);
	Py_DECREF(y_image);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_geocorrection. */

static int geoInterp2D(float x[], float y[], int n_events,
	PyArrayObject *x_image, PyArrayObject *y_image, int interp_flag,
	float origin_x, float origin_y, float xbin, float ybin) {

	/* dx and dy are the values interplated from x_image and y_image;
	   they will be subtracted from the x and y columns to correct
	   the geometric distortion.
	*/
	float dx, dy;
	int nx, ny;		/* size of images */
	int k;			/* loop index for events */
	int i, j;		/* indices in 2-D array */
	float ix, jy;		/* pixel coordinates in x_image and y_image */

	nx = PyArray_DIM(x_image, 1);	/* shape (ny,nx) */
	ny = PyArray_DIM(x_image, 0);
	if (nx != PyArray_DIM(y_image, 1) || ny != PyArray_DIM(y_image, 0)) {
	    PyErr_SetString(PyExc_RuntimeError,
		"x_image and y_image are not the same shape");
	    return 1;
	}

	for (k = 0;  k < n_events;  k++) {

	    /* Adjust for offset and scale of geo images. */
	    ix = (x[k] - origin_x) / xbin;
	    jy = (y[k] - origin_y) / ybin;

	    if (interp_flag) {

		if (ix <= -0.5 || ix >= nx-0.5 || jy <= -0.5 || jy >= ny-0.5)
		    continue;
		bilinearInterp(ix, jy, x_image, y_image, nx, ny, &dx, &dy);

	    } else {

		i = NINT(ix);
		j = NINT(jy);
		if (i < 0 || i >= nx || j < 0 || j >= ny)
		    continue;
		dx = *(float *)PyArray_GETPTR2(x_image, j, i);
		dy = *(float *)PyArray_GETPTR2(y_image, j, i);
	    }

	    /* Update x and y in-place. */
	    x[k] -= dx;
	    y[k] -= dy;
	}

	return 0;
}

/* calling sequence for walkcorrection:

   walkcorrection(x, y, image, delta)

    x, y       i: arrays of pixel coordinates of the events (float32)
    image       i: the 2-D image array of delta values (float32)
    delta:     io: array of lookups in the image at the x, y coordinates
   ccos_walkcorrection calls bilinearinterpolation, which calculates the
   walk correction.
*/

static PyObject *ccos_walkcorrection(PyObject *self, PyObject *args) {

        PyObject *ox, *oy, *o_image, *o_delta;
	PyArrayObject *x, *y, *image, *delta;
	int status;
	int n_events;		/* number of rows in events table */

	if (!PyArg_ParseTuple(args, "OOOO",
			      &ox, &oy, &o_image, &o_delta)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_FLOAT32,
			NPY_IN_ARRAY);
	y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_FLOAT32,
			NPY_IN_ARRAY);
	image = (PyArrayObject *)PyArray_FROM_OTF(o_image, NPY_FLOAT32,
			NPY_IN_ARRAY);
	delta = (PyArrayObject *)PyArray_FROM_OTF(o_delta, NPY_FLOAT32,
			NPY_INOUT_ARRAY);
	if (x == NULL || y == NULL || image == NULL || delta == NULL)
	    return NULL;

	n_events = PyArray_DIM(x, 0);	/* rows in events table */
	status = bilinearinterpolation(
		(float *)PyArray_DATA(x), (float *)PyArray_DATA(y), n_events,
		image, (float *)PyArray_DATA(delta));

	Py_DECREF(x);
	Py_DECREF(y);
	Py_DECREF(image);
	Py_DECREF(delta);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_walkcorrection. */

static int bilinearinterpolation(float x[], float y[], int n_events,
				 PyArrayObject *image, float delta[]) {

	/* delta is the array of values interpolated from image;
	   they will be subtracted from the x and y columns to correct
	   the geometric distortion.
	*/

	int nx, ny;		/* size of images */
	int k;			/* loop index for events */
	float ix, jy;		/* pixel coordinates in x_image and y_image */
	float p, q;
	float r, s;
	int i, j;

	nx = PyArray_DIM(image, 1);	/* shape (ny,nx) */
	ny = PyArray_DIM(image, 0);

	for (k = 0;  k < n_events;  k++) {
	        ix = x[k];
	        jy = y[k];
	        if (ix <= -0.5 || ix >= nx-0.5 || jy <= -0.5 || jy >= ny-0.5)
	                continue;
	        i = (int)floor((double)ix);
	        j = (int)floor((double)jy);

	        if (i < 0)
	            i = 0;
	        if (i > nx-2)
	            i = nx-2;
	        if (j < 0)
	            j = 0;
	        if (j > ny-2)
	            j = ny-2;

	        /* weights for X direction */
	        q = ix - (float)i;
	        p = 1.0F - q;

	        /* weights for Y direction */
	        s = jy - (float)j;
	        r = 1.0F - s;

	        delta[k] = p * r * *(float *)PyArray_GETPTR2(image, j, i) +
	          q * r * *(float *)PyArray_GETPTR2(image, j, i+1) +
	          p * s * *(float *)PyArray_GETPTR2(image, j+1, i) +
	          q * s * *(float *)PyArray_GETPTR2(image, j+1, i+1);

	}
	return 0;
}

/* calling sequence for pha_check:

   counters = pha_check(x, y, pha, dq, im_low, im_high, pha_flag)

    x, y        i: arrays of pixel coordinates of the events (float32)
    pha         i: array of pulse heights (int16)
    dq         io: array of data quality flags (int16)
    im_low      i: the 2-D image array of lower limits for pulse height (int16)
    im_high     i: the 2-D image array of upper limits for pulse height (int16)
    pha_flag    i: the flag value that indicates that the pulse height is
                   out of bounds (int)

    (nlow, nhigh) o: a two-element tuple giving the number of events that
                     were flagged as out of range on the low side or on
                     the high side respectively

   ccos_pha_check calls phaCheck, which compares the value of each value
   in the pha column with the lower and upper limits of the acceptable
   range for pulse height at the corresponding location on the detector.
   Values that are out of range will be flagged in the dq array.
*/

static PyObject *ccos_pha_check(PyObject *self, PyObject *args) {

	PyObject *ox, *oy, *opha, *odq, *oim_low, *oim_high;
        int pha_flag;
	PyArrayObject *x, *y, *pha, *dq, *im_low, *im_high;
	int status;
	int n_events;		/* number of rows in events table */
	/* number of events flagged because pha is below or above the cutoff */
	int nlow, nhigh;
	PyObject *counters;

	if (!PyArg_ParseTuple(args, "OOOOOOi",
			&ox, &oy, &opha, &odq, &oim_low, &oim_high,
			&pha_flag)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_FLOAT32,
			NPY_IN_ARRAY);
	y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_FLOAT32,
			NPY_IN_ARRAY);
	pha = (PyArrayObject *)PyArray_FROM_OTF(opha, NPY_INT16,
			NPY_IN_ARRAY);
	dq = (PyArrayObject *)PyArray_FROM_OTF(odq, NPY_INT16,
			NPY_INOUT_ARRAY);
	im_low = (PyArrayObject *)PyArray_FROM_OTF(oim_low, NPY_INT16,
			NPY_IN_ARRAY);
	im_high = (PyArrayObject *)PyArray_FROM_OTF(oim_high, NPY_INT16,
			NPY_IN_ARRAY);
	if (x == NULL || y == NULL || pha == NULL || dq == NULL ||
		im_low == NULL || im_high == NULL)
	    return NULL;

	n_events = PyArray_DIM(x, 0);	/* rows in events table */
	status = phaCheck(n_events, pha_flag,
		(float *)PyArray_DATA(x), (float *)PyArray_DATA(y),
		(short *)PyArray_DATA(pha), (short *)PyArray_DATA(dq),
		im_low, im_high, &nlow, &nhigh);

	Py_DECREF(x);
	Py_DECREF(y);
	Py_DECREF(pha);
	Py_DECREF(dq);
	Py_DECREF(im_low);
	Py_DECREF(im_high);

	if (status) {
	    return NULL;
	} else {
	    counters = Py_BuildValue("(i,i)", nlow, nhigh);
	    return counters;
	}
}

/* This is called by ccos_pha_check. */

static int phaCheck(int n_events, short pha_flag,
	float x[], float y[], short pha[], short dq[],
	PyArrayObject *im_low, PyArrayObject *im_high,
	int *nlow, int *nhigh) {

	int nx, ny;		/* size of images */
	int k;			/* loop index for events */
	int i, j;		/* indices in 2-D array */

	nx = PyArray_DIM(im_low, 1);	/* shape (ny,nx) */
	ny = PyArray_DIM(im_low, 0);
	if (nx != PyArray_DIM(im_high, 1) || ny != PyArray_DIM(im_high, 0)) {
	    PyErr_SetString(PyExc_RuntimeError,
		"im_low and im_high are not the same shape");
	    return 1;
	}
	*nlow = 0;
	*nhigh = 0;

	for (k = 0;  k < n_events;  k++) {

	    i = NINT(x[k]);
	    j = NINT(y[k]);
	    /* pixels outside the image array will not be checked */
	    if (i < 0 || i >= nx || j < 0 || j >= ny)
		continue;

	    /* Compare the pulse height with the lower and upper cutoffs
	       at the current location, and flag it as bad if it's out
	       of range.
	    */
	    if (pha[k] < *(short *)PyArray_GETPTR2(im_low, j, i)) {
		dq[k] |= pha_flag;		/* update dq in-place */
		(*nlow)++;
	    }
	    if (pha[k] > *(short *)PyArray_GETPTR2(im_high, j, i)) {
		dq[k] |= pha_flag;
		(*nhigh)++;
	    }
	}

	return 0;
}

/* This routine does bilinear interpolation at x,y within the x and y
   image arrays.  The interpolated values are returned as dx and dy.
   x is the more rapidly varying axis.  nx and ny give the size of the
   image arrays.
*/

static void bilinearInterp(float x, float y,
		PyArrayObject *x_image, PyArrayObject *y_image,
		int nx, int ny, float *dx, float *dy) {

	int i, j;
	float p, q, r, s;	/* 1-D weights */

	/* default values */
	*dx = 0.;
	*dy = 0.;

	i = (int)floor(x);
	j = (int)floor(y);

	if (i < 0)
	    i = 0;
	if (i > nx-2)
	    i = nx-2;
	if (j < 0)
	    j = 0;
	if (j > ny-2)
	    j = ny-2;

	/* weights for X direction */
	q = x - (float)i;
	p = 1.0F - q;

	/* weights for Y direction */
	s = y - (float)j;
	r = 1.0F - s;

	*dx = p * r * *(float *)PyArray_GETPTR2(x_image, j, i) +
	      q * r * *(float *)PyArray_GETPTR2(x_image, j, i+1) +
	      p * s * *(float *)PyArray_GETPTR2(x_image, j+1, i) +
	      q * s * *(float *)PyArray_GETPTR2(x_image, j+1, i+1);

	*dy = p * r * *(float *)PyArray_GETPTR2(y_image, j, i) +
	      q * r * *(float *)PyArray_GETPTR2(y_image, j, i+1) +
	      p * s * *(float *)PyArray_GETPTR2(y_image, j+1, i) +
	      q * s * *(float *)PyArray_GETPTR2(y_image, j+1, i+1);
}

/* calling sequence for clear_rows:

   clear_rows(dq, y_lower, y_upper, x_left, x_right)

    dq             io: array of data quality flags (int16)
    y_lower         i: array of Y coordinates at lower edge (float32)
    y_upper         i: array of Y coordinates at upper edge (float32)
    x_left          i: array of X coordinates at left edge (float32)
    x_right         i: array of X coordinates at right edge (float32)

   ccos_clear_rows calls clearRows, which assigns zero to the region within
   the specified (curved) borders.
*/

static PyObject *ccos_clear_rows(PyObject *self, PyObject *args) {

	PyObject *odq, *oy_lower, *oy_upper, *ox_left, *ox_right;
	PyArrayObject *dq, *y_lower, *y_upper, *x_left, *x_right;
	int status;

	if (!PyArg_ParseTuple(args, "OOOOO",
			&odq, &oy_lower, &oy_upper, &ox_left, &ox_right)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	dq = (PyArrayObject *)PyArray_FROM_OTF(odq, NPY_INT16,
			NPY_INOUT_ARRAY);
	y_lower = (PyArrayObject *)PyArray_FROM_OTF(oy_lower, NPY_FLOAT32,
			NPY_IN_ARRAY);
	y_upper = (PyArrayObject *)PyArray_FROM_OTF(oy_upper, NPY_FLOAT32,
			NPY_IN_ARRAY);
	x_left  = (PyArrayObject *)PyArray_FROM_OTF(ox_left, NPY_FLOAT32,
			NPY_IN_ARRAY);
	x_right = (PyArrayObject *)PyArray_FROM_OTF(ox_right, NPY_FLOAT32,
			NPY_IN_ARRAY);
	if (dq == NULL ||
	    y_lower == NULL || y_upper == NULL ||
	    x_left  == NULL || x_right == NULL)
	    return NULL;

	status = clearRows(dq,
		(float *)PyArray_DATA(y_lower),
		(float *)PyArray_DATA(y_upper),
		(float *)PyArray_DATA(x_left),
		(float *)PyArray_DATA(x_right));

	Py_DECREF(dq);
	Py_DECREF(y_lower);
	Py_DECREF(y_upper);
	Py_DECREF(x_left);
	Py_DECREF(x_right);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_clear_rows. */

static int clearRows(PyArrayObject *dq,
	float y_lower[], float y_upper[], float x_left[], float x_right[]) {

	int nx, ny;		/* size of DQ array */
	int i, j;		/* indices in DQ array */
	float ymin, ymax;	/* min of y_lower, max of y_upper */
	int iymin, iymax;	/* int ymin and ymax for loop indices */
	int *i_x_left, *i_x_right, *i_y_lower, *i_y_upper;

	nx = PyArray_DIM(dq, 1);	/* shape (ny,nx) */
	ny = PyArray_DIM(dq, 0);

	i_x_left  = PyMem_Malloc(ny * sizeof(int));
	i_x_right = PyMem_Malloc(ny * sizeof(int));
	i_y_lower = PyMem_Malloc(nx * sizeof(int));
	i_y_upper = PyMem_Malloc(nx * sizeof(int));
	if (i_x_left == NULL || i_x_right == NULL ||
	    i_y_lower == NULL || i_y_upper == NULL) {
	    PyErr_NoMemory();
	    return 1;
	}
	/* Copy to integer arrays, and compress the range in X. */
	for (j = 0;  j < ny;  j++) {
	    i_x_left[j] = (int)(ceil(x_left[j]));
	    i_x_right[j] = (int)(floor(x_right[j]));
	}
	for (i = 0;  i < nx;  i++) {
	    i_y_lower[i] = NINT(y_lower[i]);
	    i_y_upper[i] = NINT(y_upper[i]);
	}

	/* Find the min of y_lower and the max of y_upper. */
	ymin = y_lower[0];
	ymax = y_upper[0];
	for (i = 0;  i < nx;  i++) {
	    ymin = (y_lower[i] < ymin) ? y_lower[i] : ymin;
	    ymax = (y_upper[i] > ymax) ? y_upper[i] : ymax;
	}
	/* Convert to integer, and compress the range. */
	iymin = (int)(ceil(ymin));
	iymax = (int)(floor(ymax));

	for (j = iymin;  j <= iymax;  j++) {
	    for (i = i_x_left[j];  i <= i_x_right[j];  i++) {
		if (i < 0 || i >= nx)
		    continue;
		if (j >= i_y_lower[i] && j <= i_y_upper[i]) {
		    *(short *)PyArray_GETPTR2(dq, j, i) = 0;
		}
	    }
	}

	PyMem_Free(i_x_left);
	PyMem_Free(i_x_right);
	PyMem_Free(i_y_lower);
	PyMem_Free(i_y_upper);

	return 0;
}

/* calling sequence for interp1d:

   interp1d(x_a, y_a, x_b, y_b)

    x_a, y_a   i: input independent and dependent variable arrays
    x_b        i: independent variable array
    y_b       io: interpolated data at each element of x_b (modified in-place)

   ccos_interp1d calls interp1d to interpolate within y_a for each element
   of x_b.  All arrays must be 1-D.  x_a and y_a must be the same length,
   and x_b and y_b must be the same length (possibly different from the
   length of x_a).  The arrays will be converted internally to double
   if they aren't already.
*/

static PyObject *ccos_interp1d(PyObject *self, PyObject *args) {

	PyObject *ox_a, *oy_a, *ox_b, *oy_b;
	PyArrayObject *x_a, *y_a, *x_b, *y_b;
	int n_a, n_b;
	int status;

	if (!PyArg_ParseTuple(args, "OOOO",
			&ox_a, &oy_a, &ox_b, &oy_b)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	x_a = (PyArrayObject *)PyArray_FROM_OTF(ox_a, NPY_FLOAT64,
		NPY_IN_ARRAY);
	y_a = (PyArrayObject *)PyArray_FROM_OTF(oy_a, NPY_FLOAT64,
		NPY_IN_ARRAY);
	x_b = (PyArrayObject *)PyArray_FROM_OTF(ox_b, NPY_FLOAT64,
		NPY_IN_ARRAY);
	y_b = (PyArrayObject *)PyArray_FROM_OTF(oy_b, NPY_FLOAT64,
		NPY_ARRAY_INOUT_ARRAY2);
	if (x_a == NULL || y_a == NULL || x_b == NULL || y_b == NULL)
	    return NULL;

	status = interp_check(x_a, y_a, x_b, y_b);
	if (status) {
	    Py_DECREF(x_a); Py_DECREF(y_a); Py_DECREF(x_b); Py_DECREF(y_b);
	    return NULL;
	}

	n_a = PyArray_DIM(x_a, 0);
	n_b = PyArray_DIM(x_b, 0);
	interp1d((double *)PyArray_DATA(x_a),
		 (double *)PyArray_DATA(y_a), n_a,
		 (double *)PyArray_DATA(x_b),
		 (double *)PyArray_DATA(y_b), n_b);

	Py_DECREF(x_a);
	Py_DECREF(y_a);
	Py_DECREF(x_b);
	PyArray_ResolveWritebackIfCopy(y_b);
	Py_DECREF(y_b);

	Py_INCREF(Py_None);
	return Py_None;
}

/* This function compares the shapes of the arrays.  x_a and y_a must be
   the same length, and x_b and y_b must be the same length (which need not
   be the same as the length of x_a).  All arrays must be 1-D.
*/

static int interp_check(PyArrayObject *x_a, PyArrayObject *y_a,
		        PyArrayObject *x_b, PyArrayObject *y_b) {

	int n_a, n_b;

	n_a = PyArray_DIM(x_a, 0);
	n_b = PyArray_DIM(x_b, 0);
	if (n_a < 1) {
	    PyErr_SetString(PyExc_RuntimeError,
			"no data in input independent variable array");
	    return 1;
	}

	if (n_a != PyArray_DIM(y_a, 0) || n_b != PyArray_DIM(y_b, 0)) {
	    PyErr_SetString(PyExc_RuntimeError,
			"arrays have inconsistent shapes");
	    return 1;
	}
	if (PyArray_NDIM(x_a) != 1 || PyArray_NDIM(x_b) != 1) {
	    PyErr_SetString(PyExc_RuntimeError,
			"arrays must all be 1-D");
	    return 1;
	}

	return 0;
}

/* This function does linear interpolation to assign values to y_b.
   For values of x_a that are outside the range of x_b, the first
   or last value of y_a will be assigned to y_b.
*/

static void interp1d(double x_a[], double y_a[], int n_a,
		      double x_b[], double y_b[], int n_b) {

/* arguments:
double x_a[]       i: input independent variable array
double y_a[]       i: input dependent variable array
int n_a            i: size of x_a and y_a (at least 1)
double x_b[]       i: independent variable array
double y_b[]      io: interpolated value at each element of x_b
int n_b            i: size of x_b and y_b
*/

	int i_a;	/* index in x_a or y_a */
	int i_b;	/* index in x_b or y_b */
	double p, q;	/* weights for linear interpolation */

	if (n_a == 1) {

	    for (i_b = 0;  i_b < n_b;  i_b++)
		y_b[i_b] = y_a[0];

	} else {

	    for (i_b = 0;  i_b < n_b;  i_b++) {

		i_a = binarySearch(x_b[i_b], x_a, n_a);

		/* extrapolate with first or last value, if out of bounds */
		if (i_a == -1) {
		    y_b[i_b] = y_a[0];
		} else if (i_a == n_a) {
		    y_b[i_b] = y_a[n_a-1];
		} else {
		    q = (x_b[i_b] - x_a[i_a]) / (x_a[i_a+1] - x_a[i_a]);
		    p = 1. - q;
		    y_b[i_b] = p * y_a[i_a] + q * y_a[i_a+1];
		}
	    }
	}
}

/* calling sequence for getstartstop:

   getstartstop(time, istart, istop, delta_t)

    time          i: array of times (seconds) of the events (float32)
    istart        o: array of indices in events list of the start of
                     time intervals (int32)
    istop         o: array of indices of the end of time intervals (int32)
    delta_t       i: length (seconds) of each time interval (double)

   ccos_getstartstop gets the indices in the events list of the start and
   stop times of the time intervals.  istart[i] and istop[i] are intended
   to be used as the limits of a Python slice for time interval i.
*/

static PyObject *ccos_getstartstop(PyObject *self, PyObject *args) {

	PyObject *otime, *oistart, *oistop;
	PyArrayObject *time, *istart, *istop;
	double delta_t;
	int status;

	int n_events;		/* size of input arrays (number of events) */
	int nbins;		/* length of istart and istop arrays */

	if (!PyArg_ParseTuple(args, "OOOd",
			&otime, &oistart, &oistop, &delta_t)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	time = (PyArrayObject *)PyArray_FROM_OTF(otime, NPY_FLOAT32,
		NPY_IN_ARRAY);
	istart = (PyArrayObject *)PyArray_FROM_OTF(oistart, NPY_INT32,
		NPY_INOUT_ARRAY);
	istop = (PyArrayObject *)PyArray_FROM_OTF(oistop, NPY_INT32,
		NPY_INOUT_ARRAY);
	if (time == NULL || istart == NULL || istop == NULL)
	    return NULL;

	n_events = PyArray_DIM(time, 0);
	nbins = PyArray_DIM(istart, 0);

	status = getStartStopTimes((float *)PyArray_DATA(time), n_events,
		(int *)PyArray_DATA(istart), (int *)PyArray_DATA(istop),
		nbins, delta_t);

	Py_DECREF(time);
	Py_DECREF(istart);
	Py_DECREF(istop);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This function, called by ccos_getstartstop, finds the indices in the
   events list of the start and stop times of the time intervals (nbins
   such intervals, uniformly spaced in time, starting at time[0]).
*/

static int getStartStopTimes(float time[], int n_events,
		int istart[], int istop[],
		int nbins, double delta_t) {

/* arguments:
time			i: time at each event
n_events                i: number of events (length of time array)
istart, istop		o: arrays of start and stop event numbers
nbins                   i: length of istart, istop
delta_t			i: length of time interval
*/

	int i;			/* index for istart, istop */
	int k;			/* index for events */
	double begin_interval;	/* time of beginning of a delta_t interval */
	double end_interval;	/* time at end of a delta_t interval */
	double end_next_interval;	/* time at end of following interval */
	int done;

	/* Initialize.  The last element of istop will not otherwise be
	   assigned (unless the time range is divisible by delta_t), so
	   initialize istop to n_events.
	*/
	for (i = 0;  i < nbins;  i++) {		/* initialize */
	    istart[i] = 0;
	    istop[i] = n_events;
	}

	/* Fill in the start and stop index of each time interval.
	   The istart and istop values can be used as limits of a Python
	   slice, e.g. time[istart[i]:istop[i]].
	*/
	k = 0;
	istart[0] = 0;
	for (i = 0;  i < nbins;  i++) {

	    if (k >= n_events) {
		istart[i] = n_events;
		istop[i] = n_events;
		continue;
	    }

	    begin_interval = time[0] + i * delta_t;
	    end_interval = begin_interval + delta_t;
	    end_next_interval = begin_interval + 2. * delta_t;

	    done = 0;
	    while (!done) {
		if (k >= n_events) {
		    break;
		}
		if (time[k] >= begin_interval) {
		    istart[i] = k;
		    if (time[k] > end_interval) {	/* empty interval */
			istop[i] = k;
			done = 1;
		    }
		    break;
		}
		k++;
	    }

	    while (!done) {
		if (k >= n_events) {
		    break;
		}
		if (time[k] >= end_interval) {
		    if (time[k] >= end_next_interval && k > 0) {
			istop[i] = k - 1;
		    } else {
			istop[i] = k;
		    }
		    break;
		} else {
		    k++;
		}
	    }
	}

	return 0;
}

/* calling sequence for getbkgcounts:

   getbkgcounts(y, dq,
	istart, istop, bkg_counts, src_counts,
	bkg1_low, bkg1_high, bkg2_low, bkg2_high,
	src_low, src_high, bkgsf)

    y             i: array of Y pixel coordinates of the events
                     (either float32 or int16)
    dq            i: array of data quality flags for the events (int16)
    istart        i: array of indices in events list of the start of
                     time intervals (int32)
    istop         i: array of indices of the end of time intervals (int32)
    bkg_counts    o: array of background counts in each interval (int32)
    src_counts    o: array of source counts in each interval (int32)
    bkg1_low,     i: row numbers of lower background region (int)
      bkg1_high
    bkg2_low,     i: row numbers of upper background region (int)
      bkg2_high
    src_low,      i: row numbers of source region (int)
      src_high
    bkgsf         i: scale factor to estimate how many background counts
                     there are in the source extraction region (double)

   ccos_getbkgcounts gets the number of source and background counts within
   each interval [istart[i]:istop[i]].
*/

static PyObject *ccos_getbkgcounts(PyObject *self, PyObject *args) {

	PyObject *oy, *odq, *oistart, *oistop,
		*obkg_counts, *osrc_counts;
	PyArrayObject *y, *dq, *istart, *istop,
		*bkg_counts, *src_counts;
	int bkg1_low, bkg1_high, bkg2_low, bkg2_high;
	int src_low, src_high;
	double bkgsf;
	int status;

	int n_events;		/* length of y and dq arrays */
	int nbins;		/* length of bkg_counts (and other) arrays */

	if (!PyArg_ParseTuple(args, "OOOOOOiiiiiid",
			&oy, &odq, &oistart, &oistop,
			&obkg_counts, &osrc_counts,
			&bkg1_low, &bkg1_high, &bkg2_low, &bkg2_high,
			&src_low, &src_high, &bkgsf)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	if (PyArray_TYPE(oy) == NPY_INT16) {
	    y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	dq = (PyArrayObject *)PyArray_FROM_OTF(odq, NPY_INT16, NPY_IN_ARRAY);
	istart = (PyArrayObject *)PyArray_FROM_OTF(oistart, NPY_INT32,
		NPY_IN_ARRAY);
	istop = (PyArrayObject *)PyArray_FROM_OTF(oistop, NPY_INT32,
		NPY_IN_ARRAY);
	bkg_counts = (PyArrayObject *)PyArray_FROM_OTF(obkg_counts, NPY_INT32,
		NPY_INOUT_ARRAY);
	src_counts = (PyArrayObject *)PyArray_FROM_OTF(osrc_counts, NPY_INT32,
		NPY_INOUT_ARRAY);
	if (y == NULL || dq == NULL || istart == NULL || istop == NULL ||
		bkg_counts == NULL || src_counts == NULL)
	    return NULL;

	n_events = PyArray_DIM(y, 0);
	nbins = PyArray_DIM(bkg_counts, 0);

	status = getBkgCounts(y, (short *)PyArray_DATA(dq), n_events,
		(int *)PyArray_DATA(istart), (int *)PyArray_DATA(istop),
		(int *)PyArray_DATA(bkg_counts),
		(int *)PyArray_DATA(src_counts), nbins,
		bkg1_low, bkg1_high, bkg2_low, bkg2_high,
		src_low, src_high, bkgsf);

	Py_DECREF(y);
	Py_DECREF(dq);
	Py_DECREF(istart);
	Py_DECREF(istop);
	Py_DECREF(bkg_counts);
	Py_DECREF(src_counts);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This function, called by ccos_getbkgcounts, finds the number of
   source and background counts within each interval.
*/

static int getBkgCounts(PyArrayObject *y, short dq[], int n_events,
		int istart[], int istop[],
		int bkg_counts[], int src_counts[],
		int nbins,
		int bkg1_low, int bkg1_high, int bkg2_low, int bkg2_high,
		int src_low, int src_high, double bkgsf) {

/* arguments:
y			i: y location of each event
dq			i: data quality flag for each event
n_events                i: length of y and dq arrays
istart, istop		i: arrays of start and stop event numbers
bkg_counts		o: array of background counts
src_counts		o: array of source counts
nbins                   i: length of istart, istop, bkg_counts, src_counts
bkg1_low, bkg1_high	i: background region, low and high (inclusive) rows
bkg2_low, bkg2_high	i: background region, low and high (inclusive)
src_low, src_high	i: source region, low and high (inclusive)
bkgsf 			i: background scale factor
*/

	int y_type;		/* data type code */
	int i;			/* index for istart, istop */
	int k;			/* index for events */
	int jy;			/* y rounded to an int */
	int n_src, n_bkg;	/* counters for events within interval */
	/* individual values */
	float c_y;

	y_type = y->descr->type_num;

	/* Fill in the values for the number of source and background
	   counts within each time interval.
	*/
	for (i = 0;  i < nbins;  i++) {
	    if (istart[i] > n_events || istop[i] > n_events) {
		PyErr_SetString(PyExc_RuntimeError,
			"value of istart or istop is too large");
		return 1;
	    }
	    n_src = 0;
	    n_bkg = 0;
	    for (k = istart[i];  k < istop[i];  k++) {
		if (dq[k] == 0) {
		    if (y_type == NPY_INT16) {
			jy = *(short *)PyArray_GETPTR1(y, k);
		    } else {
			c_y = *(float *)PyArray_GETPTR1(y, k);
			jy = NINT(c_y);
		    }
		    if (jy >= src_low && jy <= src_high) {
			n_src++;		/* within source region */
		    } else if ((jy >= bkg1_low && jy <= bkg1_high) ||
			       (jy >= bkg2_low && jy <= bkg2_high)) {
			n_bkg++;		/* within background region */
		    }
		}
	    }
	    bkg_counts[i] = n_bkg;
	    /* Correct the source counts for the number of background
		counts expected within the source region, assuming uniform
		background.
	    */
	    src_counts[i] = n_src - n_bkg * bkgsf;
	}
	return 0;
}

/* calling sequence for smallerbursts:

   smallerbursts(time, dq,
	istart, istop, bkg_counts, src_counts,
	smallest_burst, stdrej, source_frac,
	half_block, max_iter,
	large_burst, small_burst, dq_burst, verbose)

    time            i: array of times, used only if verbose (float32)
    dq             io: array of data quality flags for the events (int16)
    istart          i: array of indices in events list of the start of
                       time intervals (int32)
    istop           i: array of indices of the end of time intervals (int32)
    bkg_counts     io: array of background counts in each interval (int32)
    src_counts      i: array of source counts in each interval (int32)
    delta_t         i: length (seconds) of each time interval (double)
    smallest_burst  i: burst_min * delta_t (double)
    stdrej          i: criterion for N * sigma rejection (double)
    source_frac     i: minimum fraction of source counts (double)
    half_block      i: round (median_dt / delta_t) / 2 (int)
    max_iter        i: maximum number of iterations (int)
    large_burst     i: flag value for a "large" burst (int)
    small_burst     i: flag value for a "small" burst (int)
    dq_burst        i: data quality flag for a burst (int)
    verbose         i: if true, print info about bursts (int)

   ccos_smallerbursts screens for smaller bursts.  If any are found,
   they will be flagged by setting elements of the dq array to dq_burst,
   and the bkg_counts element for each burst will be set to small_burst
   (assumed to be negative).
*/

static PyObject *ccos_smallerbursts(PyObject *self, PyObject *args) {

	PyObject *otime, *odq, *oistart, *oistop, *obkg_counts, *osrc_counts;
	PyArrayObject *time, *dq, *istart, *istop, *bkg_counts, *src_counts;
	double delta_t, smallest_burst, stdrej, source_frac;
	int half_block, max_iter,
		large_burst, small_burst, dq_burst, verbose;
	int n_events;		/* length of y and dq arrays */
	int nbins;		/* length of bkg_counts (and other) arrays */
	int status;

	if (!PyArg_ParseTuple(args, "OOOOOOddddiiiiii",
			&otime, &odq, &oistart, &oistop,
			&obkg_counts, &osrc_counts,
			&delta_t, &smallest_burst, &stdrej, &source_frac,
			&half_block, &max_iter,
			&large_burst, &small_burst, &dq_burst, &verbose)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	time = (PyArrayObject *)PyArray_FROM_OTF(otime, NPY_FLOAT32,
		NPY_IN_ARRAY);
	dq = (PyArrayObject *)PyArray_FROM_OTF(odq, NPY_INT16,
		NPY_INOUT_ARRAY);
	istart = (PyArrayObject *)PyArray_FROM_OTF(oistart, NPY_INT32,
		NPY_IN_ARRAY);
	istop = (PyArrayObject *)PyArray_FROM_OTF(oistop, NPY_INT32,
		NPY_IN_ARRAY);
	bkg_counts = (PyArrayObject *)PyArray_FROM_OTF(obkg_counts, NPY_INT32,
		NPY_INOUT_ARRAY);
	src_counts = (PyArrayObject *)PyArray_FROM_OTF(osrc_counts, NPY_INT32,
		NPY_IN_ARRAY);
	if (time == NULL || dq == NULL || istart == NULL || istop == NULL ||
		bkg_counts == NULL || src_counts == NULL)
	    return NULL;

	n_events = PyArray_DIM(dq, 0);
	nbins = PyArray_DIM(bkg_counts, 0);

	status = findSmallerBursts((float *)PyArray_DATA(time),
		(short *)PyArray_DATA(dq), n_events,
		(int *)PyArray_DATA(istart), (int *)PyArray_DATA(istop),
		(int *)PyArray_DATA(bkg_counts),
		(int *)PyArray_DATA(src_counts), nbins, delta_t,
		smallest_burst, stdrej, source_frac,
		half_block, max_iter,
		large_burst, small_burst, dq_burst, verbose);

	Py_DECREF(time);
	Py_DECREF(dq);
	Py_DECREF(istart);
	Py_DECREF(istop);
	Py_DECREF(bkg_counts);
	Py_DECREF(src_counts);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This function, called by ccos_smallerbursts, screens for "smaller"
   bursts.  This is done iteratively, up to max_iter times.  Within each
   iteration, bkg_counts is boxcar filtered (into a scratch array), but
   using median within the box.  The box size is 2 * half_block + 1, but
   it is truncated on one side as an endpoint is approached.  Negative
   values in bkg_counts are used to flag bursts, so when computing the
   median, values less than zero are ignored.

   The rejection criteria are based on the difference delta_counts between
   an element of bkg_counts and the filtered value.  For element i to be
   flagged as a burst, the following three criteria must all be met:

	delta_counts > smallest_burst
	delta_counts > stdrej * sqrt(bkg_counts[i])
	delta_counts > source_frac * src_counts[i]
*/

static int findSmallerBursts(float time[], short dq[], int n_events,
	int istart[], int istop[],
	int bkg_counts[], int src_counts[], int nbins, double delta_t,
	double smallest_burst, double stdrej, double source_frac,
	int half_block, int max_iter,
	int large_burst, int small_burst, int dq_burst, int verbose) {

/*
arguments:
time			 i: time at each event
dq			io: data quality flag for each event
n_events                 i: size of time and dq arrays
istart, istop		 i: arrays of start and stop event numbers
bkg_counts		io: array of background counts
src_counts		 i: array of source counts
nbins			 i: size of istart, istop, bkg_counts, src_counts
delta_t			 i: length (seconds) of each time interval
smallest_burst		 i: burst_min * delta_t
stdrej			 i: criterion for N * sigma rejection
source_frac		 i: minimum fraction of source counts
half_block		 i: half the box size for boxcar median smoothing
max_iter		 i: maximum number of iterations
large_burst		 i: flag value for a "large" burst
small_burst		 i: flag value for a "small" burst
dq_burst		 i: data quality flag for a burst
verbose			 i: if true, print info about bursts
*/

	int *m_filtered;	/* median filtered bkg_counts */
	int i, k, iter;		/* loop indices */
	int nreject;		/* counter for number of intervals rejected */
	float c_time;		/* value in time array */
	short c_dq;		/* value in dq array */
	/* difference between bkg_counts and median filtered bkg_counts */
	int delta_counts;

	m_filtered = PyMem_Malloc(nbins * sizeof(int));
	if (m_filtered == NULL)
	    return 1;

	for (iter = 0;  iter < max_iter;  iter++) {

	    nreject = 0;
	    if (median_boxcar(bkg_counts, m_filtered,
			nbins, half_block, large_burst))
		return 1;

	    for (i = 0;  i < nbins;  i++) {
		if (istart[i] > n_events || istop[i] > n_events) {
		    PyErr_SetString(PyExc_RuntimeError,
			"value of istart or istop is too large");
		    return 1;
		}
		delta_counts = bkg_counts[i] - m_filtered[i];
		if (bkg_counts[i] > 0 &&
		    delta_counts > smallest_burst &&
		    delta_counts > stdrej * sqrt((double)bkg_counts[i]) &&
		    delta_counts > source_frac * src_counts[i]) {

		    nreject++;
		    if (verbose) {
			c_time = time[istart[i]];
			printf("burst at time %d, counts = %d,"
				" median = %d, diff = %d, source = %d\n",
				(int)(c_time + delta_t/2.),
				bkg_counts[i], m_filtered[i],
				delta_counts, src_counts[i]);
		    }
		    for (k = istart[i];  k <= istop[i];  k++) {
			c_dq = (short)dq_burst;
			dq[k] |= c_dq;
		    }
		    bkg_counts[i] = small_burst;
		}
	    }
	    if (verbose) {
		if (nreject < 1) {
		    if (iter == 0)
			printf("No small burst detected.\n");
		    else
			printf("No further bursts detected after"
				" iteration %d.\n", iter+1);
		} else {
		    printf("After iteration %d, we found %d intervals"
			" affected by bursts.\n", iter+1, nreject);
		}
	    }
	    if (nreject < 1)
		break;
	}
	PyMem_Free(m_filtered);

	return 0;
}

/* Boxcar smooth bkg_counts, with a box of width (2 * half_block + 1).
   Within the box, the smoothed value is the median of all non-negative
   values (previously detected bursts will have been flagged by replacing
   the value in bkg_counts with a negative number).  As an endpoint is
   approached, the box will be truncated on the side of the endpoint.

   The result is returned in the array m_filtered.
*/

static int median_boxcar(int bkg_counts[], int m_filtered[],
		int nbins, int half_block, int large_burst) {

/*
arguments:
bkg_counts		i: array of background counts
m_filtered		o: median filtered array of background counts
nbins                   i: size of bkg_counts and m_filtered
half_block		i: the boxcar size is 2 * half_block + 1
large_burst		i: flag value for a "large" burst
*/

	int *temp;	/* scratch for a copy of the data to be sorted */
	int i0, i1;	/* range of indices of copied data */
	int lenblk;	/* length of copied data */
	int i;
	int base;	/* index of first non-negative element of sorted data */
	int mid;	/* index of midpoint of non-negative data */

	temp = PyMem_Malloc((2*half_block+1) * sizeof(int));
	if (temp == NULL)
	    return 1;

	for (i = 0;  i < nbins;  i++) {
	    /* Shrink the filter near the endpoints. */
	    i0 = i - half_block;
	    if (i0 < 0) i0 = 0;
	    i1 = i + half_block;
	    if (i1 >= nbins) i1 = nbins-1;
	    lenblk = i1 - i0 + 1;

	    memcpy(temp, bkg_counts+i0, lenblk*sizeof(int));
	    qsort(temp, lenblk, sizeof(int), compare_int);
	    /* Ignore values less than 0 (used to flag rejected elements). */
	    for (base = 0;  base < lenblk;  base++) {
		if (temp[base] >= 0)
		    break;
	    }
	    if (base >= lenblk-1) {
		m_filtered[i] = large_burst;
	    } else {
		mid = (lenblk - 1 - base) / 2;
		m_filtered[i] = temp[base+mid];
	    }
	}
	PyMem_Free(temp);
	return 0;
}

static int compare_int(const void *vp, const void *vq) {

	const int *p = vp;
	const int *q = vq;

	if (*p > *q)
	    return 1;
	else if (*p < *q)
	    return -1;
	else
	    return 0;
}

/* calling sequence for getbadtime:

   badtime = getbadtime(time, dq)

    time       i: array of times of the events (float32)
    dq         i: array of data quality flags for the events (int16)

   The function value is the sum of the time intervals within which
   the data quality flag was non-zero for every event in the intervals.
*/

static PyObject *ccos_getbadtime(PyObject *self, PyObject *args) {

	PyObject *otime, *odq;
	PyArrayObject *time, *dq;
	double badtime;
	int n_events;

	if (!PyArg_ParseTuple(args, "OO", &otime, &odq)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	time = (PyArrayObject *)PyArray_FROM_OTF(otime, NPY_FLOAT32,
		NPY_IN_ARRAY);
	dq = (PyArrayObject *)PyArray_FROM_OTF(odq, NPY_INT16, NPY_IN_ARRAY);
	if (time == NULL || dq == NULL)
	    return NULL;

	n_events = PyArray_DIM(dq, 0);

	badtime = getBadTime((float *)PyArray_DATA(time),
		(short *)PyArray_DATA(dq), n_events);

	Py_DECREF(time);
	Py_DECREF(dq);

	return Py_BuildValue("d", badtime);	/* return badtime */
}

static double getBadTime(float time[], short dq[], int n_events) {

	double badtime;		/* sum of bad time intervals */
	int in_bad_interval;	/* true if we're in a flagged interval */
	int k;
	double c_t0, c_t1;	/* times at limits of a bad time interval */

	badtime = 0.;
	in_bad_interval = 0;
	c_t0 = 0.;		/* initialization shouldn't be necessary */
	for (k = 0;  k < n_events;  k++) {
	    if (dq[k] != 0) {
		if (!in_bad_interval) {
		    in_bad_interval = 1;
		    c_t0 = time[k];
		}
	    } else if (in_bad_interval) {
		in_bad_interval = 0;			/* dq[k] = 0 */
		/* end of interval is previous time */
		c_t1 = time[k-1];
		badtime += (c_t1 - c_t0);
	    }
	}
	if (in_bad_interval)
	    badtime += (time[n_events-1] - c_t0);

	return badtime;
}

/* calling sequence for xy_extract:

   xy_extract(xi, eta, outdata, slope, intercept, x_offset,
               dq, sdqflags, epsilon)

    xi, eta     i: arrays of pixel coordinates of the events
                   (either float32 or int16); xi is in the dispersion
                   direction, eta is cross-dispersion
    outdata    io: a 2-D array, into which the extracted spectrum will be put
                   (float64)
    slope       i: the slope of the band (pixels per pixel, double)
    intercept   i: the zero point of the band (pixel number, double)
    x_offset    i: the offset (it's zero or positive) to add to the xi pixel
                   coordinate to get the pixel in the output array (int)

   optional arguments:
    dq          i: array of data quality flags (int16)
    sdqflags    i: bit mask for the "serious" dq flags (short)
    epsilon     i: array of weights for the events (float32)

   xi and eta may be either Float32 or Int16, and they do not need to be
   the same type.

   The x_offset argument is the offset to add to values from xi to get the
   index into the output spectrum.  This lets the output spectrum be longer
   than the detector width.

   slope is the slope of the band with respect to the axis along which
   it will be extracted.  If slope=+0.1, for example, then the band is
   about six degrees counterclockwise from the horizontal axis (or
   clockwise from the vertical axis for NUV).

   ccos_xy_extract calls extrFromEvents, which first initializes the output
   spectrum to zero and then increments elements of the spectrum, one for
   each event within the spectral extraction region.  The output array is
   2-D, as with ccos_extractband; sum along axis 0 to get a 1-D spectrum.

   The location of the region to be extracted is specified by slope and
   intercept.  The slope is in pixels per pixel.  The intercept is zero
   indexed, at the eta = 0 edge; this is the location where the center line
   of the region to be extracted meets the edge.  No interpolation is done.
*/

static PyObject *ccos_xy_extract(PyObject *self, PyObject *args) {

	PyObject *oxi, *oeta, *ooutdata, *odq, *oepsilon;
	PyArrayObject *xi, *eta, *outdata, *dq, *epsilon;
	double slope, intercept;
	int x_offset;
	short sdqflags;
	int status;

	x_offset = 0;
	odq = NULL;
	oepsilon = NULL;
	sdqflags = 0;

	if (!PyArg_ParseTuple(args, "OOOddi|OhO",
			&oxi, &oeta, &ooutdata,
			&slope, &intercept, &x_offset,
			&odq, &sdqflags, &oepsilon)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	if (PyArray_TYPE(oxi) == NPY_INT16) {
	    xi = (PyArrayObject *)PyArray_FROM_OTF(oxi, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    xi = (PyArrayObject *)PyArray_FROM_OTF(oxi, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	if (PyArray_TYPE(oeta) == NPY_INT16) {
	    eta = (PyArrayObject *)PyArray_FROM_OTF(oeta, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    eta = (PyArrayObject *)PyArray_FROM_OTF(oeta, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	if (xi == NULL || eta == NULL)
	    return NULL;

	outdata = (PyArrayObject *)PyArray_FROM_OTF(ooutdata, NPY_FLOAT64,
		NPY_INOUT_ARRAY);
	if (outdata == NULL)
	    return NULL;

	if (odq == NULL) {
	    dq = NULL;
	} else {
	    dq = (PyArrayObject *)PyArray_FROM_OTF(odq, NPY_INT16,
			NPY_IN_ARRAY);
	    if (dq == NULL)
		return NULL;
	}
	if (oepsilon == NULL) {
	    epsilon = NULL;
	} else {
	    epsilon = (PyArrayObject *)PyArray_FROM_OTF(oepsilon, NPY_FLOAT32,
			NPY_IN_ARRAY);
	    if (epsilon == NULL)
		return NULL;
	}

	status = extrFromEvents(xi, eta, outdata,
			x_offset, slope, intercept,
			dq, sdqflags, epsilon);

	Py_DECREF(xi);
	Py_DECREF(eta);
	Py_DECREF(outdata);
	Py_XDECREF(dq);
	Py_XDECREF(epsilon);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_xy_extract. */

static int extrFromEvents(PyArrayObject *xi, PyArrayObject *eta,
		PyArrayObject *outdata,
		int x_offset, double slope, double intercept,
		PyArrayObject *dq, short sdqflags, PyArrayObject *epsilon) {

	int xi_type, eta_type;	/* data type code for xi and eta */
	int half_height;	/* half of extr height, fraction truncated */
	/* lower is the lower limit of the spectral extraction region for a
	   given value of xi.
	*/
	double y0;		/* lower edge of spec extr region at xi=0 */
	double y;		/* c_eta corrected for slope */
	int n_events;		/* length of xi and eta arrays */
	int k;			/* event index */
	int nx, ny;		/* size of outdata */
	double c_xi, c_eta;	/* xi and eta for one event */
	short c_dq = 0;
	double c_eps = 1.;	/* but the epsilon column is float32 */
	int i, j;		/* nearest integers to c_xi, c_eta */
	int i_z;		/* i + zero-point offset */

	n_events = PyArray_DIM(xi, 0);
	if (n_events != PyArray_DIM(eta, 0)) {
	    PyErr_SetString(PyExc_RuntimeError,
			"xi and eta must both be the same length");
	    return 1;
	}

	xi_type = xi->descr->type_num;
	eta_type = eta->descr->type_num;

	/* shape is (ny,nx), nx is in the dispersion direction */
	nx = PyArray_DIM(outdata, 1);
	ny = PyArray_DIM(outdata, 0);

	half_height = ny / 2;			/* truncate */

	/* Initialize outdata to zero, because we're going to increment
	   a pixel value for each event in the list.
	*/
	for (i = 0;  i < nx;  i++)
	    for (j = 0;  j < ny;  j++)
		*(double *)PyArray_GETPTR2(outdata, j, i) = 0.;

	y0 = intercept - half_height;
	for (k = 0;  k < n_events;  k++) {	/* for each event ... */
	    if (dq != NULL)
		c_dq = *(short *)PyArray_GETPTR1(dq, k);
	    if ((c_dq & sdqflags) == 0) {
		if (xi_type == NPY_INT16) {
		    i = *(short *)PyArray_GETPTR1(xi, k);
		    c_xi = (double)i;
		} else {
		    c_xi = *(float *)PyArray_GETPTR1(xi, k);
		    i = NINT(c_xi);
		}
		i_z = i + x_offset;	/* note:  don't add x_offset to c_xi */
		if (i_z < 0 || i_z > nx-1)
		    continue;
		if (eta_type == NPY_INT16) {
		    j = *(short *)PyArray_GETPTR1(eta, k);
		    c_eta = (double)j;
		} else {
		    c_eta = *(float *)PyArray_GETPTR1(eta, k);
		}
		y = c_eta - (y0 + slope * c_xi);
		j = NINT(y);
		/* include this event if it's within the extraction region */
		if (j >= 0 && j < ny) {
		    if (epsilon != NULL)
			c_eps = *(float *)PyArray_GETPTR1(epsilon, k);
		    *(double *)PyArray_GETPTR2(outdata, j, i_z) += c_eps;
		}
	    }
	}

	return (0);
}

/* calling sequence for xy_collapse:

   xy_collapse(xi, eta, dq, slope, xdisp)

    xi, eta     i: arrays of pixel coordinates of the events
                   (either float32 or int16); xi is in the dispersion
                   direction, eta is cross-dispersion
    dq          i: array of data quality flags (int16)
    slope       i: the slope of the band (pixels per pixel, double)
    xdisp      io: a 1-D array, into which the collapsed data will be put;
                   the location of a feature in this array shows where the
                   spectrum crosses the left edge of the detector (float64)

   xi and eta may be either Float32 or Int16, and they do not need to be
   the same type.

   slope is the slope (in pixels per pixel) of the band with respect to
   the axis along which it will be extracted.  See the description for
   xy_extract.

   ccos_xy_collapse calls collapseFromEvents, which first initializes
   xdisp to zero, then collapses the data along the dispersion direction,
   incrementing xdisp for each event.  The length of xdisp should be
   the width of the detector in the cross-dispersion direction.  If the
   slope of the spectrum were zero, element i of xdisp would be incremented
   for each element of eta that is between i-0.5 and i+0.5.  For a non-zero
   slope, each eta position is first adjusted by subtracting (slope * xi),
   so the position is projected to the left edge.  Note that this is the same
   convention as for the intercept as given in the xtractab reference table.
*/

static PyObject *ccos_xy_collapse(PyObject *self, PyObject *args) {

	PyObject *oxi, *oeta, *odq, *oxdisp;
	PyArrayObject *xi, *eta, *dq, *xdisp;
	double slope;
	int status;
	int n_events;		/* length of xi, eta, dq arrays */
	int length;		/* length of xdisp array */

	if (!PyArg_ParseTuple(args, "OOOdO",
			&oxi, &oeta, &odq, &slope, &oxdisp)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	if (PyArray_TYPE(oxi) == NPY_INT16) {
	    xi = (PyArrayObject *)PyArray_FROM_OTF(oxi, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    xi = (PyArrayObject *)PyArray_FROM_OTF(oxi, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	if (PyArray_TYPE(oeta) == NPY_INT16) {
	    eta = (PyArrayObject *)PyArray_FROM_OTF(oeta, NPY_INT16,
		NPY_IN_ARRAY);
	} else {
	    eta = (PyArrayObject *)PyArray_FROM_OTF(oeta, NPY_FLOAT32,
		NPY_IN_ARRAY);
	}
	dq = (PyArrayObject *)PyArray_FROM_OTF(odq, NPY_INT16, NPY_IN_ARRAY);
	xdisp = (PyArrayObject *)PyArray_FROM_OTF(oxdisp, NPY_FLOAT64,
		NPY_ARRAY_INOUT_ARRAY2);
	if (xi == NULL || eta == NULL || dq == NULL || xdisp == NULL)
	    return NULL;

	n_events = PyArray_DIM(xi, 0);
	if (n_events != PyArray_DIM(eta, 0) ||
	    n_events != PyArray_DIM(dq, 0)) {
	    PyErr_SetString(PyExc_RuntimeError,
			"xi, eta and dq must all be the same length");
	    return NULL;
	}
	length = PyArray_DIM(xdisp, 0);
	status = collapseFromEvents(xi, eta,
		(short *)PyArray_DATA(dq), n_events,
		slope,
		(double *)PyArray_DATA(xdisp), length);

	Py_DECREF(xi);
	Py_DECREF(eta);
	Py_DECREF(dq);
	PyArray_ResolveWritebackIfCopy(xdisp);
	Py_DECREF(xdisp);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF(Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_xy_collapse. */

static int collapseFromEvents(PyArrayObject *xi, PyArrayObject *eta,
		short dq[], int n_events,
		double slope,
		double xdisp[], int length) {

	int xi_type, eta_type;	/* data type code for xi and eta */
	int k;			/* event index */
	/* xi and eta for one event */
	double c_xi, c_eta;
	int i, j;		/* nearest integers to c_xi, c_eta */

	xi_type = xi->descr->type_num;
	eta_type = eta->descr->type_num;

	for (i = 0;  i < length;  i++)
	    xdisp[i] = 0.;

	for (k = 0;  k < n_events;  k++) {
	    if (dq[k] == 0) {
		if (xi_type == NPY_INT16) {
		    i = *(short *)PyArray_GETPTR1(xi, k);
		    c_xi = (double)i;
		} else {
		    c_xi = *(float *)PyArray_GETPTR1(xi, k);
		}
		if (eta_type == NPY_INT16) {
		    j = *(short *)PyArray_GETPTR1(eta, k);
		    c_eta = (double)j;
		} else {
		    c_eta = *(float *)PyArray_GETPTR1(eta, k);
		}
		/* shift to where spectrum crosses left edge */
		c_eta -= slope * c_xi;
		j = NINT(c_eta);
		if (j >= 0 && j < length)
		    xdisp[j] += 1.;
	    }
	}

	return (0);
}

/* calling sequence for csum_3d:

   csum_3d(array, x, y, epsilon, pha, binx, biny)

    array     io: the output 3-D array (float32)
    x, y       i: arrays of pixel coordinates of the events (float32)
    epsilon    i: array of weights for the events (float32)
    pha        i: array of pulse heights (int16)

   optional arguments:
    binx       i: binning factor in the more rapidly varying axis
    biny       i: binning factor in the less rapidly varying axis

   ccos_csum_3d calls bin3DtoCsum, which converts arrays of pixel
   coordinates to an image array.  The 3-D array ('array') is assumed
   to have already been initialized to zero.  For each event n, the
   array element at [pha[n],y[n],x[n]] will be incremented by epsilon[n].
*/

static PyObject *ccos_csum_3d(PyObject *self, PyObject *args) {

	PyObject *oarray, *ox, *oy, *oepsilon, *opha;
	PyArrayObject *array, *x, *y, *epsilon, *pha;
	int binx=1, biny=1;
	int n_events, nx, ny, nz;

	if (!PyArg_ParseTuple(args, "OOOOO|ii",
			&oarray, &ox, &oy, &oepsilon, &opha, &binx, &biny)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	array = (PyArrayObject *)PyArray_FROM_OTF(oarray, NPY_FLOAT32,
			NPY_INOUT_ARRAY);
	if (array == NULL)
	    return NULL;

	x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_FLOAT32, NPY_IN_ARRAY);
	y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_FLOAT32, NPY_IN_ARRAY);

	epsilon = (PyArrayObject *)PyArray_FROM_OTF(oepsilon, NPY_FLOAT32,
			NPY_IN_ARRAY);
	pha = (PyArrayObject *)PyArray_FROM_OTF(opha, NPY_INT16, NPY_IN_ARRAY);

	if (x == NULL || y == NULL || epsilon == NULL || pha == NULL)
	    return NULL;

	n_events = PyArray_DIM(x, 0);

	if (PyArray_NDIM(array) == 3) {
	    nx = PyArray_DIM(array, 2);		/* shape (nz,ny,nx) */
	    ny = PyArray_DIM(array, 1);
	    nz = PyArray_DIM(array, 0);
	} else if (PyArray_NDIM(array) == 2) {
	    nx = PyArray_DIM(array, 1);		/* shape (ny,nx) */
	    ny = PyArray_DIM(array, 0);
	    nz = 1;
	} else {
	    PyErr_SetString(PyExc_RuntimeError,
		"the array must be either 2-D or 3-D");
	    return NULL;
	}

	bin3DtoCsum((float *)PyArray_DATA(array), nx, ny, nz,
		binx, biny,
		(float *)PyArray_DATA(x), (float *)PyArray_DATA(y),
		(float *)PyArray_DATA(epsilon),
		(short *)PyArray_DATA(pha), n_events);

	Py_DECREF(array);
	Py_DECREF(x);
	Py_DECREF(y);
	Py_DECREF(epsilon);
	Py_DECREF(pha);

	Py_INCREF(Py_None);
	return Py_None;
}

/* This is called by ccos_csum_3d. */

static void bin3DtoCsum(float array[], int nx, int ny, int nz,
		int binx, int biny,
		float x[], float y[],
		float epsilon[], short pha[], int n_events) {

	int n;		/* loop index for events */
	int i, j, k;	/* pixel coordinates of event, indices in 3-D array */
	int m;		/* 1-D index into array */

	if (binx < 1)
	    binx = 1;
	if (biny < 1)
	    biny = 1;

	for (n = 0;  n < n_events;  n++) {

	    /* the pixel coordinates of the current event */
	    i = NINT(x[n]) / binx;
	    j = NINT(y[n]) / biny;
	    k = pha[n];

	    /* truncate at borders of image */
	    if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz)
		continue;

	    m = i + nx*j + nx*ny*k;
	    array[m] += epsilon[n];
	}
}

/* calling sequence for csum_2d:

   csum_2d(array, x, y, epsilon, binx, biny)

    array     io: the output 2-D array (float32)
    x, y       i: arrays of pixel coordinates of the events (float32)
    epsilon    i: array of weights for the events (float32)

   optional arguments:
    binx       i: binning factor in the more rapidly varying axis
    biny       i: binning factor in the less rapidly varying axis

   ccos_csum_2d calls bin2DtoCsum, which converts arrays of pixel
   coordinates to an image array.  The 2-D array ('array') is assumed
   to have already been initialized to zero.  For each event n, the
   array element at [y[n],x[n]] will be incremented by epsilon[n].
*/

static PyObject *ccos_csum_2d(PyObject *self, PyObject *args) {

	PyObject *oarray, *ox, *oy, *oepsilon;
	PyArrayObject *array, *x, *y, *epsilon;
	int binx=1, biny=1;
	int n_events, nx, ny;

	if (!PyArg_ParseTuple(args, "OOOO|ii",
			&oarray, &ox, &oy, &oepsilon, &binx, &biny)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	array = (PyArrayObject *)PyArray_FROM_OTF(oarray, NPY_FLOAT32,
			NPY_INOUT_ARRAY);
	if (array == NULL)
	    return NULL;

	x = (PyArrayObject *)PyArray_FROM_OTF(ox, NPY_FLOAT32, NPY_IN_ARRAY);
	y = (PyArrayObject *)PyArray_FROM_OTF(oy, NPY_FLOAT32, NPY_IN_ARRAY);

	epsilon = (PyArrayObject *)PyArray_FROM_OTF(oepsilon, NPY_FLOAT32,
			NPY_IN_ARRAY);

	if (x == NULL || y == NULL || epsilon == NULL)
	    return NULL;

	n_events = PyArray_DIM(x, 0);
	nx = PyArray_DIM(array, 1);	/* shape (ny,nx) */
	ny = PyArray_DIM(array, 0);

	bin2DtoCsum((float *)PyArray_DATA(array), nx, ny,
		binx, biny,
		(float *)PyArray_DATA(x), (float *)PyArray_DATA(y),
		(float *)PyArray_DATA(epsilon), n_events);

	Py_DECREF(array);
	Py_DECREF(x);
	Py_DECREF(y);
	Py_DECREF(epsilon);

	Py_INCREF(Py_None);
	return Py_None;
}

/* This is called by ccos_csum_2d. */

static void bin2DtoCsum(float array[], int nx, int ny,
		int binx, int biny,
		float x[], float y[],
		float epsilon[], int n_events) {

	int n;		/* loop index for events */
	int i, j;	/* pixel coordinates of event, indices in 2-D array */

	if (binx < 1)
	    binx = 1;
	if (biny < 1)
	    biny = 1;

	for (n = 0;  n < n_events;  n++) {

	    /* the pixel coordinates of the current event */
	    i = NINT(x[n]) / binx;
	    j = NINT(y[n]) / biny;

	    /* truncate at borders of image */
	    if (i < 0 || i >= nx || j < 0 || j >= ny)
		continue;

	    array[i+nx*j] += epsilon[n];
	}
}

/* calling sequence for bin2d:

   bin2d(array, binned_array)

    array        i: the input 2-D array (float32)
    binned_array o: the output 2-D array (float32)

   ccos_bin2d calls bin2DArray, which bins the input 'array' by integer
   factors in each axis and writes the results to the output 'binned_array'.
   The binning factor must be an integer in each axis; that is, the shape
   of 'binned_array' must divide the shape of 'array'.
*/

static PyObject *ccos_bin2d(PyObject *self, PyObject *args) {

	PyObject *oarray, *obinned_array;
	PyArrayObject *array, *binned_array;
	int nx, ny, nxb, nyb;

	if (!PyArg_ParseTuple(args, "OO", &oarray, &obinned_array)) {
	    PyErr_SetString(PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	array = (PyArrayObject *)PyArray_FROM_OTF(oarray,
			NPY_FLOAT32, NPY_IN_ARRAY);
	binned_array = (PyArrayObject *)PyArray_FROM_OTF(obinned_array,
			NPY_FLOAT32, NPY_INOUT_ARRAY);
	if (array == NULL || binned_array == NULL)
	    return NULL;

	nx = PyArray_DIM(array, 1);	/* shape (ny,nx) */
	ny = PyArray_DIM(array, 0);
	nxb = PyArray_DIM(binned_array, 1);
	nyb = PyArray_DIM(binned_array, 0);
	if (nx / nxb * nxb != nx || ny / nyb * nyb != ny) {
	    PyErr_SetString(PyExc_RuntimeError, "bin factors must be integer");
	    return NULL;
	}

	bin2DArray((float *)PyArray_DATA(array), nx, ny,
		   (float *)PyArray_DATA(binned_array), nxb, nyb);

	Py_DECREF(array);
	Py_DECREF(binned_array);

	Py_INCREF(Py_None);
	return Py_None;
}

/* This is called by ccos_bin2d. */

static void bin2DArray(float array[], int nx, int ny,
		float binned_array[], int nxb, int nyb) {

	int binx, biny;
	int i, j;	/* indices in array */
	int ib, jb;	/* indices in binned_array */

	binx = nx / nxb;
	biny = ny / nyb;

	for (i = 0;  i < nxb*nyb;  i++)
	    binned_array[i] = 0.;

	for (j = 0;  j < ny;  j++) {
	    jb = j / biny;
	    for (i = 0;  i < nx;  i++) {
		ib = i / binx;
		binned_array[ib+nxb*jb] += array[i+nx*j];
	    }
	}
}

static PyMethodDef ccos_methods[] = {
	{"binevents", ccos_binevents, METH_VARARGS,
	"bin events table x & y coordinates to an image array"},

	{"bindq", ccos_bindq, METH_VARARGS,
	"flag regions in a 2-D array according to a DQI table"},

	{"applydq", ccos_applydq, METH_VARARGS,
	"assign data quality flags from a DQI table into an events table column"},

	{"dq_or", ccos_dq_or, METH_VARARGS,
	"bitwise OR each column of a 2-D array, writing to a 1-D array"},

	{"applyflat", ccos_applyflat, METH_VARARGS,
	"divide events table EPSILON column by a flat field"},

	{"range", ccos_range, METH_VARARGS,
	"return (i0, i1) such that all values in time[i0:i1] are within the range from t0 to t1"},

	{"unbinaccum", ccos_unbinaccum, METH_VARARGS,
	"convert an image array to a pseudo time-tag list"},

	{"addrandom", ccos_addrandom, METH_VARARGS,
	"add a pseudo-random number between -0.5 and +0.5 to each element of x"},

	{"convolve1d", ccos_convolve1d, METH_VARARGS,
	"convolve a 2-D array with a 1-D array"},

	{"extractband", ccos_extractband, METH_VARARGS,
	"copy out a 2-D band (spectrum or background) from a 2-D array"},

	{"smoothbkg", ccos_smoothbkg, METH_VARARGS,
	"boxcar smooth a 1-D array"},

	{"addlines", ccos_addlines, METH_VARARGS,
	"create a template spectrum based on a list of emission lines"},

	{"geocorrection", ccos_geocorrection, METH_VARARGS,
	"apply the geometric correction to events table x and y arrays"},

	{"walkcorrection", ccos_walkcorrection, METH_VARARGS,
	 "calculate the walk correction for x or y event tables"},

	{"pha_check", ccos_pha_check, METH_VARARGS,
	"apply the pulse height correction to events table pha array"},

	{"clear_rows", ccos_clear_rows, METH_VARARGS,
	"assign 0 to the region in a DQ array within specified borders"},

	{"interp1d", ccos_interp1d, METH_VARARGS,
	"linearly interpolate within y_a for each element of x_b"},

	{"getstartstop", ccos_getstartstop, METH_VARARGS,
	    "get indices at start & stop of each time interval"},

	{"getbkgcounts", ccos_getbkgcounts, METH_VARARGS,
	    "get source and background counts"},

	{"smallerbursts", ccos_smallerbursts, METH_VARARGS,
	    "screen for smaller bursts"},

	{"getbadtime", ccos_getbadtime, METH_VARARGS,
	    "return sum of time intervals flagged in dq"},

	{"xy_extract", ccos_xy_extract, METH_VARARGS,
	    "extract a 1-D spectrum from events list"},

	{"xy_collapse", ccos_xy_collapse, METH_VARARGS,
	    "collapse an events list along the dispersion direction"},

	{"csum_3d", ccos_csum_3d, METH_VARARGS,
	    "bin 3-D events to a 'calcos sum' (csum) image"},

	{"csum_2d", ccos_csum_2d, METH_VARARGS,
	    "bin 2-D events to a 'calcos sum' (csum) image"},

	{"bin2d", ccos_bin2d, METH_VARARGS,
	    "bin (block sum) a 2-D array to a smaller 2-D array"},

	{NULL, NULL, 0, NULL}
};

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ccos",
    NULL,
    -1,
    ccos_methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if defined(NPY_PY3K)
PyObject *PyInit_ccos(void) 
#else
PyMODINIT_FUNC initccos(void)
#endif
{
	PyObject *mod;		/* the module */
	PyObject *dict;		/* the module's dictionary */

#if defined(NPY_PY3K)
	mod = PyModule_Create(&moduledef);
#else
	mod = Py_InitModule("ccos", ccos_methods);
#endif
	import_array();

	/* set the doc string */
	dict = PyModule_GetDict(mod);
#if defined(NPY_PY3K)
	PyDict_SetItemString(dict, "__doc__",
		PyUnicode_FromString(DocString()));
        return mod;
#else
	PyDict_SetItemString(dict, "__doc__",
		PyString_FromString(DocString()));
	return;
#endif
}
