/* This module contains the following functions:

binevents bins a list of (x,y) coordinates into a 2-D array.
bindq updates a 2-D array of data quality flags from DQI table info.
applydq assigns data quality flags from a DQI table into a column.
applyflat divides the epsilon values by a flat field.
range returns indices for a slice of time.
unbinaccum updates X & Y coordinates of pixel values in an image.
addrandom adds pseudo-random numbers on (-0.5,+0.5) to values in an array.
convolve1d convolves a 2-D image with a 1-D array.
extractband extracts a 2-D band (spectrum or background) from a 2-D image.
smoothbkg smooths a 1-D array (background).
addlines creates a template spectrum based on a list of emission lines.
geocorrection applies the geometric (INL) distortion correction.
interp1d does linear interpolation of one 1-D array onto another.

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
*/

# include <Python.h>

# include <stdlib.h>
# include <string.h>
# include <math.h>
# include <limits.h>
# include <time.h>

# include "numpy/libnumarray.h"

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
# define NINT(x)  ((int) (floor (x + 0.5)))

static char *DocString (void);

static PyObject *ccos_binevents (PyObject *, PyObject *);
static PyObject *ccos_bindq (PyObject *, PyObject *);
static PyObject *ccos_applydq (PyObject *, PyObject *);
static PyObject *ccos_applyflat (PyObject *, PyObject *);
static PyObject *ccos_range (PyObject *, PyObject *);
static PyObject *ccos_unbinaccum (PyObject *, PyObject *);
static PyObject *ccos_addrandom (PyObject *, PyObject *);
static PyObject *ccos_convolve1d (PyObject *, PyObject *);
static PyObject *ccos_extractband (PyObject *, PyObject *);
static PyObject *ccos_smoothbkg (PyObject *, PyObject *);
static PyObject *ccos_addlines (PyObject *, PyObject *);
static PyObject *ccos_geocorrection (PyObject *, PyObject *);
static PyObject *ccos_interp1d (PyObject *, PyObject *);
static PyObject *ccos_getstartstop (PyObject *self, PyObject *args);
static PyObject *ccos_getbkgcounts (PyObject *, PyObject *);
static PyObject *ccos_smallerbursts (PyObject *, PyObject *);
static PyObject *ccos_getbadtime (PyObject *, PyObject *);
static PyObject *ccos_xy_extract (PyObject *, PyObject *);
static PyObject *ccos_xy_collapse (PyObject *, PyObject *);

static int binEventsToImage (PyArrayObject *, PyArrayObject *,
	PyArrayObject *, PyArrayObject *, short, PyArrayObject *);
static int binDQToImage (PyArrayObject *, PyArrayObject *,
	PyArrayObject *, PyArrayObject *, PyArrayObject *, PyArrayObject *,
	int, int, int);
static int applyDQToEvents (PyArrayObject *, PyArrayObject *,
		PyArrayObject *, PyArrayObject *, PyArrayObject *,
		PyArrayObject *, PyArrayObject *, PyArrayObject *);

static int applyFlatField (PyArrayObject *, PyArrayObject *,
	PyArrayObject *, PyArrayObject *, int, int);
static PyObject *timeRange (PyArrayObject *, double, double);
static int search (PyArrayObject *, int, float);
static int search_d (PyArrayObject *, int, double);
static int unbinImage (PyArrayObject *, PyArrayObject *, PyArrayObject *);
static PyObject *addRN (PyArrayObject *, int, int);
static int convolveWithDopp (PyArrayObject *, PyArrayObject *, int);
static int extract2DBand (PyArrayObject *,
	int, double, double, PyArrayObject *);
static int smoothBackground (PyArrayObject *, int);
static int addEmissionLines (PyArrayObject *, PyArrayObject *,
	double, PyArrayObject *, PyArrayObject *);
static double findPixelNumber (double, double [], int);
static int binarySearch (double, double [], int);
static void addLSF (double, float, double, float *, int);
static int geoInterp2D (PyArrayObject *, PyArrayObject *,
	PyArrayObject *, PyArrayObject *, int, float, float, float, float);
static void bilinearInterp (float, float,
	PyArrayObject *, PyArrayObject *, int, int,
	float *, float *);
static int interp_check (PyArrayObject *, PyArrayObject *,
			 PyArrayObject *, PyArrayObject *);
static int interp1d (double [], double [], int, double [], double [], int);
static int getStartStopTimes (PyArrayObject *, PyArrayObject *y,
		PyArrayObject *,
		int [], int [],
		int, double);
static int getBkgCounts (PyArrayObject *, PyArrayObject *,
		int [], int [], int [], int [],
		int,
		int, int, int, int,
		int, int, double);
static int findSmallerBursts (PyArrayObject *, PyArrayObject *,
		int[], int[], int[], int[], int, double,
		double, double, double,
		int, int,
		int, int, int, int);
static int median_boxcar (int [], int [], int, int, int);
static int compare_int (const void *, const void *);
static PyObject *getBadTime (PyArrayObject *, PyArrayObject *);
static int extrFromEvents (PyArrayObject *, PyArrayObject *,
		PyArrayObject *,
		int, double, double,
		double [], int);
static int collapseFromEvents (PyArrayObject *, PyArrayObject *,
		PyArrayObject *,
		double, double [], int);

/* This function returns the documentation string to be assigned to
   __doc__.
*/

static char *DocString (void) {

	return (
"This module contains the following functions:\n\n\
    binevents (x, y, array,\n\
                <optional:  dq, epsilon>)\n\
    bindq (lx, ly, dx, dy, flag, dq_array,\n\
                <optional:  axis, mindopp, maxdopp>)\n\
    applydq (lx, ly, dx, dy, flag, x, y, dq)\n\
    applyflat (x, y, epsilon, flat,\n\
                <optional:  x_offset, y_offset>)\n\
    indices = range (time, t0, t1)\n\
    unbinaccum (image, x, y)\n\
    newseed = addrandom (x, seed, use_clock)\n\
    convolve1d (flat, dopp, axis)\n\
    extractband (indata, axis, slope, intercept, outdata)\n\
    smoothbkg (data, width)\n\
    addlines (intensity, wavelength, reswidth, x1d_wl, dq, template)\n\
    geocorrection (x, y, x_image, y_image, interp_flag,\n\
                <optional:  x_offset, y_offset, xbin, ybin>)\n\
    interp1d (x_a, y_a, x_b, y_b)\n\
    getstartstop (time, y, dq, istart, istop, delta_t)\n\
    getbkgcounts (y, dq,\n\
                istart, istop, bkg_counts, src_counts,\n\
                bkg1_low, bkg1_high, bkg2_low, bkg2_high,\n\
                src_low, src_high, bkgsf)\n\
    smallerbursts (time, dq,\n\
                istart, istop, bkg_counts, src_counts,\n\
                delta_t, smallest_burst, stdrej, source_frac,\n\
                half_block, max_iter,\n\
                large_burst, small_burst, dq_burst, verbose)\n\
    getbadtime (time, dq)\n\n\
x and y are arrays of pixel coordinates of the events (Float32 or Int16).\n\
epsilon is an array of weights for the events (Float32).\n\
dq is an array of data quality flags (0 is good; Int16).\n\
array is the 2-D array modified in-place by binevents (Float32).\n\
lx and ly are arrays of lower left corners of DQ regions (Int32).\n\
dx and dy are arrays of DQ region widths (Int32).\n\
flag is an array of data quality flags to assign to DQ regions (Int16).\n\
dq_array is the 2-D array modified in-place by bindq (Int16).\n\
mindopp and maxdopp are pixel offsets for Doppler shift (int).\n\
flat is a flat field (a 2-D array) (Float32).\n\
time is the array of times of the events (Float32 or Float64).\n\
t0, t1 is a range of times within the time array (float).\n\
indices is a two-element tuple, the limits of the slice of time (int).\n\
image is a 2-D image array to be converted to a list of pixel coordinates\n\
    (Int32, Int16, UInt16, or Float32).\n\
ncounts is the sum of the pixel values in the image (int).\n\
seed is a 32-bit integer for starting the pseudo-random number generator.\n\
newseed is the value of seed after addrandom has been called.\n\
If use_clock is true, use the system clock to generate the seed.\n\
dopp is a 1-D array with which flat will be convolved (Float32).\n\
axis (0 or 1) is the axis along which the convolution will be done (int).\n\
indata and outdata for extractband can be Int16 or Float32.\n\
For binevents, dq and epsilon are optional arguments.\n\
For bindq, axis, mindopp and maxdopp are optional arguments.\n");
}

/* calling sequence for binevents:

   binevents (x, y, array, dq, sdqflags, epsilon)

    x, y       i: arrays of pixel coordinates of the events
                  (either Float32 or Int16)
    array     io: the output 2-D array (Float32)

   optional arguments:
    dq         i: array of data quality flags (Int16; 0 is good)
    sdqflags   i: bit mask for the "serious" dq flags (short)
    epsilon    i: array of weights for the events (Float32)

   ccos_binevents calls binEventsToImage, which converts arrays of pixel
   coordinates to an image array.  The 2-D array ('array') will first be
   initialized to zero.  For each pair of elements (x[i],y[i]), the value
   in the nearest pixel of array will be incremented.  If epsilon is not
   null, the increment will be epsilon[i]; otherwise, the increment will be
   one.  If dq is not null, the pixel will be incremented only if dq[i]
   does not include a "serious" flag value (e.g. pulse height out of
   range or within a bad time interval).
*/

static PyObject *ccos_binevents (PyObject *self, PyObject *args) {

	PyObject *ox, *oy, *oarray, *odq, *oepsilon;
	PyArrayObject *x, *y, *array, *dq, *epsilon;
	short sdqflags;
	int status;

	odq = NULL;
	oepsilon = NULL;

	if (!PyArg_ParseTuple (args, "OOO|OhO",
			&ox, &oy, &oarray, &odq, &sdqflags, &oepsilon)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	x = NA_InputArray (ox, tAny, 0);
	y = NA_InputArray (oy, tAny, 0);
	array = NA_IoArray (oarray, tFloat32, 0);
	if (odq == NULL)
	    dq = NULL;
	else
	    dq = NA_InputArray (odq, tInt16, 0);
	if (oepsilon == NULL)
	    epsilon = NULL;
	else
	    epsilon = NA_InputArray (oepsilon, tFloat32, 0);

	status = binEventsToImage (x, y, array, dq, sdqflags, epsilon);

	Py_DECREF (x);
	Py_DECREF (y);
	Py_DECREF (array);
	Py_XDECREF (dq);
	Py_XDECREF (epsilon);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_binevents. */

static int binEventsToImage (PyArrayObject *x, PyArrayObject *y,
	PyArrayObject *array,
	PyArrayObject *dq, short sdqflags, PyArrayObject *epsilon) {

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

	x_type = x->descr->type_num;
	y_type = y->descr->type_num;

	/* Check the data types of the arrays. */
	if (x_type != tFloat32 && x_type != tInt16) {
	    PyErr_SetString (PyExc_RuntimeError, "x has the wrong data type");
	    return 1;
	}
	if (y_type != tFloat32 && y_type != tInt16) {
	    PyErr_SetString (PyExc_RuntimeError, "y has the wrong data type");
	    return 1;
	}
	if (array->descr->type_num != tFloat32) {
	    PyErr_SetString (PyExc_RuntimeError, "array must be Float32");
	    return 1;
	}
	if (dq != NULL && dq->descr->type_num != tInt16) {
	    PyErr_SetString (PyExc_RuntimeError, "dq must be Int16");
	    return 1;
	}
	if (epsilon != NULL && epsilon->descr->type_num != tFloat32) {
	    PyErr_SetString (PyExc_RuntimeError, "epsilon must be Float32");
	    return 1;
	}

	n_events = x->dimensions[0];
	nx = array->dimensions[1];
	ny = array->dimensions[0];

	/* Initialize array to zero, because we're going to increment
	   a pixel value for each event in the list.
	*/
	for (i = 0;  i < nx;  i++)
	    for (j = 0;  j < ny;  j++)
		NA_SET2 (array, Float32, j, i, 0.);

	for (k = 0;  k < n_events;  k++) {

	    /* get the coordinates of the current event */
	    if (x_type == tFloat32) {
		c_x = NA_GET1 (x, Float32, k);
		i = NINT (c_x);		/* the more rapidly varying index */
	    } else {
		i = NA_GET1 (x, Int16, k);
	    }
	    if (y_type == tFloat32) {
		c_y = NA_GET1 (y, Float32, k);
		j = NINT (c_y);
	    } else {
		j = NA_GET1 (y, Int16, k);
	    }

	    if (dq == NULL)
		c_dq = 0;
	    else
		c_dq = NA_GET1 (dq, Int16, k);

	    if ((c_dq & sdqflags) == 0) {

		if (epsilon == NULL)
		    c_eps = 1.;
		else
		    c_eps = NA_GET1 (epsilon, Float32, k);

		/* truncate at borders of image */
		if (i < 0 || i >= nx || j < 0 || j >= ny)
		    continue;

		c_eps += NA_GET2 (array, Float32, j, i);
		NA_SET2 (array, Float32, j, i, c_eps);
	    }
	}

	return 0;
}

/* calling sequence for bindq:

   bindq (lx, ly, dx, dy, flag, dq_array, axis, mindopp, maxdopp)

    lx, ly     i: arrays of lower left corners of regions (Int32)
    dx, dy     i: arrays of region widths (Int32)
    flag       i: array of data quality flags (Int32)
    dq_array  io: 2-D array (Int16); dq_array must have already been
                  initialized before calling bindq

   optional arguments:
    axis       i: the axis along which Doppler shift will be applied (int);
                  this is used in conjunction with mindopp and maxdopp
    mindopp    i: minimum Doppler shift (pixels) during the exposure (int)
    maxdopp    i: maximum Doppler shift (pixels) during the exposure (int)

   The arrays lx, ly, dx, dy and flag are all of the same length; these
   were taken from the data quality initialization table.

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

static PyObject *ccos_bindq (PyObject *self, PyObject *args) {

	PyObject *olx, *oly, *odx, *ody, *oflag, *odq_array;
	PyArrayObject *lx, *ly, *dx, *dy, *flag, *dq_array;
	int axis, mindopp, maxdopp;
	int status;

	axis = -1;
	mindopp = 0;
	maxdopp = 0;

	if (!PyArg_ParseTuple (args, "OOOOOO|iii",
		&olx, &oly, &odx, &ody, &oflag, &odq_array,
		&axis, &mindopp, &maxdopp)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	lx = NA_InputArray (olx, tInt32, 0);
	ly = NA_InputArray (oly, tInt32, 0);
	dx = NA_InputArray (odx, tInt32, 0);
	dy = NA_InputArray (ody, tInt32, 0);
	flag = NA_InputArray (oflag, tInt32, 0);
	dq_array = NA_IoArray (odq_array, tInt16, 0);

	status = binDQToImage (lx, ly, dx, dy, flag, dq_array,
		axis, mindopp, maxdopp);

	Py_DECREF (lx);
	Py_DECREF (ly);
	Py_DECREF (dx);
	Py_DECREF (dy);
	Py_DECREF (flag);
	Py_DECREF (dq_array);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_bindq. */

static int binDQToImage (PyArrayObject *lx, PyArrayObject *ly,
	PyArrayObject *dx, PyArrayObject *dy,
	PyArrayObject *flag, PyArrayObject *dq_array,
	int axis, int mindopp, int maxdopp) {

	int nrows;		/* number of rows in data quality table */
	int nx, ny;		/* size of array */
	int k;			/* loop index for events */
	int i, j;		/* indices in 2-D array */
	/* individual values */
	int c_dx, c_dy;		/* width of region */
	int c_lx, c_ly;		/* lower left corner */
	int c_ux, c_uy;		/* upper right corner (from lx,ly and dx, dy) */
	int c_flag;		/* flag value */
	int temp_flag;

	nrows = lx->dimensions[0];
	nx = dq_array->dimensions[1];
	ny = dq_array->dimensions[0];

	for (k = 0;  k < nrows;  k++) {

	    c_lx = NA_GET1 (lx, Int32, k);
	    c_ly = NA_GET1 (ly, Int32, k);
	    c_dx = NA_GET1 (dx, Int32, k);
	    c_dy = NA_GET1 (dy, Int32, k);
	    c_flag = NA_GET1 (flag, Int32, k);
	    c_ux = c_lx + c_dx - 1;
	    c_uy = c_ly + c_dy - 1;
	    if (axis == 1) {			/* dispaxis = 1 */
		c_lx += mindopp;
		c_ux += maxdopp;
	    } else if (axis == 0) {		/* dispaxis = 2 */
		c_ly += mindopp;
		c_uy += maxdopp;
	    }				/* else ignore Doppler shift */

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
		    temp_flag = c_flag | NA_GET2 (dq_array, Int16, j, i);
		    NA_SET2 (dq_array, Int16, j, i, temp_flag);
		}
	    }
	}

	return 0;
}

/* calling sequence for applydq:

   applydq (lx, ly, dx, dy, flag, x, y, dq)

    lx, ly     i: arrays of lower left corners of regions (Int32)
    dx, dy     i: arrays of region widths (Int32)
    flag       i: array of data quality flags (Int32)
    x, y       i: arrays of pixel coordinates of the events, not Doppler
                  corrected (either Int16 or Float32)
    dq        io: array of data quality flags (Int16)

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

static PyObject *ccos_applydq (PyObject *self, PyObject *args) {

	PyObject *olx, *oly, *odx, *ody, *oflag, *ox, *oy, *odq;
	PyArrayObject *lx, *ly, *dx, *dy, *flag, *x, *y, *dq;
	int status;

	if (!PyArg_ParseTuple (args, "OOOOOOOO",
		&olx, &oly, &odx, &ody, &oflag, &ox, &oy, &odq)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	lx = NA_InputArray (olx, tInt32, 0);
	ly = NA_InputArray (oly, tInt32, 0);
	dx = NA_InputArray (odx, tInt32, 0);
	dy = NA_InputArray (ody, tInt32, 0);
	flag = NA_InputArray (oflag, tInt32, 0);
	x = NA_InputArray (ox, tAny, 0);
	y = NA_InputArray (oy, tAny, 0);
	dq = NA_IoArray (odq, tInt16, 0);

	status = applyDQToEvents (lx, ly, dx, dy, flag, x, y, dq);

	Py_DECREF (lx);
	Py_DECREF (ly);
	Py_DECREF (dx);
	Py_DECREF (dy);
	Py_DECREF (flag);
	Py_DECREF (x);
	Py_DECREF (y);
	Py_DECREF (dq);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_applydq. */

static int applyDQToEvents (PyArrayObject *lx, PyArrayObject *ly,
		PyArrayObject *dx, PyArrayObject *dy, PyArrayObject *flag,
		PyArrayObject *x, PyArrayObject *y, PyArrayObject *dq) {

	int x_type, y_type;	/* data type codes for x and y */
	int i, j;		/* x & y rounded to int */
	int k;			/* loop index for events */
	int row, nrows;		/* loop index; number of rows in bpixtab */
	int n_events;		/* number of rows in events table */
	/* individual values */
	int c_dx, c_dy;		/* width of region */
	int *c_lx, *c_ly;	/* array of lower left corners */
	int *c_ux, *c_uy;	/* array of upper right corners */
	int *c_flag;		/* array of flag values for regions */
	float c_x;		/* rawx coordinate of an event */
	float c_y;		/* rawy coordinate of an event */
	short c_dq;		/* data quality column value */

	nrows = lx->dimensions[0];	/* rows in bpixtab */
	n_events = x->dimensions[0];	/* rows in events table */
	x_type = x->descr->type_num;
	y_type = y->descr->type_num;

	if (x_type != tInt16 && x_type != tFloat32) {
	    PyErr_SetString (PyExc_RuntimeError, "x has the wrong data type");
	    return 1;
	}
	if (y_type != tInt16 && y_type != tFloat32) {
	    PyErr_SetString (PyExc_RuntimeError, "y has the wrong data type");
	    return 1;
	}

	c_lx = PyMem_Malloc (nrows * sizeof (int));
	c_ly = PyMem_Malloc (nrows * sizeof (int));
	c_ux = PyMem_Malloc (nrows * sizeof (int));
	c_uy = PyMem_Malloc (nrows * sizeof (int));
	c_flag = PyMem_Malloc (nrows * sizeof (int));
	if (c_lx == NULL || c_ly == NULL ||
	    c_ux == NULL || c_uy == NULL || c_flag == NULL) {
	    PyErr_NoMemory();
	    return 1;
	}

	/* For each row in the data quality initialization (bpixtab)
	   reference table, get the location of the region and the
	   flag value.  Save this info for the loop over rows in the
	   events table.
	*/
	for (row = 0;  row < nrows;  row++) {

	    c_dx = NA_GET1 (dx, Int32, row);
	    c_dy = NA_GET1 (dy, Int32, row);

	    c_lx[row] = NA_GET1 (lx, Int32, row);
	    c_ly[row] = NA_GET1 (ly, Int32, row);
	    c_flag[row] = NA_GET1 (flag, Int32, row);
	    c_ux[row] = c_lx[row] + c_dx - 1;
	    c_uy[row] = c_ly[row] + c_dy - 1;
	}

	/* For each row in the events table, include the flag for each
	   region of the data quality initialization table.
	*/
	for (k = 0;  k < n_events;  k++) {

	    if (x_type == tFloat32) {
		c_x = NA_GET1 (x, Float32, k);
		i = NINT (c_x);		/* the more rapidly varying index */
	    } else {
		i = NA_GET1 (x, Int16, k);
	    }
	    if (y_type == tFloat32) {
		c_y = NA_GET1 (y, Float32, k);
		j = NINT (c_y);
	    } else {
		j = NA_GET1 (y, Int16, k);
	    }

	    /* Check each row of the bpixtab for overlap with (i,j). */
	    for (row = 0;  row < nrows;  row++) {
		if (i >= c_lx[row] && i <= c_ux[row] &&
		    j >= c_ly[row] && j <= c_uy[row]) {
		    c_dq = NA_GET1 (dq, Int16, k) | c_flag[row];
		    NA_SET1 (dq, Int16, k, c_dq);
		}
	    }
	}

	PyMem_Free (c_lx);
	PyMem_Free (c_ly);
	PyMem_Free (c_ux);
	PyMem_Free (c_uy);
	PyMem_Free (c_flag);

	return 0;
}

/* calling sequence for applyflat:

   applyflat (x, y, epsilon, flat, x_offset, y_offset)

    x, y       i: arrays of pixel coordinates of the events
                  (either Float32 or Int16)
    epsilon   io: an array of efficiencies for the events (Float32)
    flat       i: the 2-D flat field image array (Float32)

   optional arguments:
    x_offset   i: offset in the more rapidly varying axis (int)
    y_offset   i: offset in the less rapidly varying axis (int)
            x_offset and y_offset are the offsets of the flat field array
            from the beginning of the detector, to allow the flat to be
	    a subarray.  x_offset & y_offset are the values of the keywords
	    ORIGIN_X and ORIGIN_Y respectively.  These are in units of
            pixels, and they are zero-indexed.  These are the negative of
            the IRAF keywords LTV1 & LTV2 respectively.

   ccos_applyflat calls applyFlatField, which applies a flat field image to
   the epsilon (weight) array.  It is assumed that epsilon has previously
   been initialized to one, although it may subsequently have been modified,
   e.g. by applying a deadtime correction.
*/

static PyObject *ccos_applyflat (PyObject *self, PyObject *args) {

	PyObject *ox, *oy, *oepsilon, *oflat;
	int x_offset, y_offset;
	PyArrayObject *x, *y, *epsilon, *flat;
	int status;

	x_offset = 0;
	y_offset = 0;

	if (!PyArg_ParseTuple (args, "OOOO|ii",
			&ox, &oy, &oepsilon, &oflat, &x_offset, &y_offset)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	x = NA_InputArray (ox, tAny, 0);
	y = NA_InputArray (oy, tAny, 0);
	epsilon = NA_IoArray (oepsilon, tFloat32, 0);
	flat = NA_InputArray (oflat, tFloat32, 0);

	status = applyFlatField (x, y, epsilon, flat, x_offset, y_offset);

	Py_DECREF (x);
	Py_DECREF (y);
	Py_DECREF (epsilon);
	Py_DECREF (flat);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_applyflat. */

static int applyFlatField (PyArrayObject *x, PyArrayObject *y,
	PyArrayObject *epsilon, PyArrayObject *flat,
	int x_offset, int y_offset) {

	int x_type, y_type;	/* data type codes for x and y */
	int nx, ny;		/* size of flat */
	int k;			/* loop index for events */
	int n_events;		/* number of rows in events table */
	int i, j;		/* indices in 2-D array */
	/* individual values */
	float c_x, c_y, c_flat, c_eps;

	x_type = x->descr->type_num;
	y_type = y->descr->type_num;

	/* Check the data types of the arrays. */
	if (x_type != tInt16 && x_type != tFloat32) {
	    PyErr_SetString (PyExc_RuntimeError, "x has the wrong data type");
	    return 1;
	}
	if (y_type != tInt16 && y_type != tFloat32) {
	    PyErr_SetString (PyExc_RuntimeError, "y has the wrong data type");
	    return 1;
	}

	n_events = x->dimensions[0];	/* rows in events table */
	nx = flat->dimensions[1];
	ny = flat->dimensions[0];

	/* For each event, find the location in the flat field, and
	   divide the current value of epsilon by the flat field value.
	*/
	for (k = 0;  k < n_events;  k++) {

	    /* get the coordinates of the current event */
	    if (x_type == tFloat32) {
		c_x = NA_GET1 (x, Float32, k);
		i = NINT (c_x) - x_offset;	/* more rapidly varying */
	    } else {
		i = NA_GET1 (x, Int16, k) - x_offset;
	    }
	    if (y_type == tFloat32) {
		c_y = NA_GET1 (y, Float32, k);
		j = NINT (c_y) - y_offset;
	    } else {
		j = NA_GET1 (y, Int16, k) - y_offset;
	    }

	    /* ignore events that are outside the flat field image */
	    if (i < 0 || i >= nx || j < 0 || j >= ny)
		continue;

	    c_flat = NA_GET2 (flat, Float32, j, i);
	    if (c_flat > 0.) {
		c_eps = NA_GET1 (epsilon, Float32, k);
		c_eps /= c_flat;
		NA_SET1 (epsilon, Float32, k, c_eps);
	    }
	}

	return 0;
}

/* calling sequence for range:

   indices = range (time, t0, t1)

    time       i: array of times of the events (either Float32 or Float64)
    t0, t1     i: the times for which the indices are needed (double)

    indices    o: a two-element tuple of the indices in the time array
                  that bracket t0 and t1, to be used in the sense:
                      for i in range (indices[0], indices[1]):

   ccos_range calls timeRange, which returns (i0, i1) such that all values
   in time[i0:i1] are within the range from t0 to t1, and that would not
   be true for a smaller i0 or a larger i1.  time is assumed to be
   monotonically nondecreasing.
*/

static PyObject *ccos_range (PyObject *self, PyObject *args) {

	PyObject *otime;
	double t0, t1;
	PyArrayObject *time;
	PyObject *indices;

	if (!PyArg_ParseTuple (args, "Odd", &otime, &t0, &t1)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	time = NA_InputArray (otime, tAny, 0);

	indices = timeRange (time, t0, t1);

	Py_DECREF (time);

	return indices;
}

/* This is called by ccos_range. */

static PyObject *timeRange (PyArrayObject *time, double t0, double t1) {

	int i0, i1;
	int time_type;		/* data type code for time */
	double temp;		/* for swapping t0 & t1, if out of order */
	double tfirst, tlast;	/* first and last times in time */
	int n_events;		/* length of time array */
	PyObject *indices;

	/* Check that the data type of the input time is supported. */
	time_type = time->descr->type_num;
	if (time_type != tFloat32 && time_type != tFloat64) {
	    PyErr_SetString (PyExc_RuntimeError,
			"data type of time is not supported");
	    return NULL;
	}

	if (t1 < t0) {			/* swap, if out of order */
	    temp = t0;
	    t0 = t1;
	    t1 = temp;
	}

	n_events = time->dimensions[0];

	/* get times of first and last events */
	if (time_type == tFloat32) {
	    tfirst = NA_GET1 (time, Float32, 0);
	    tlast = NA_GET1 (time, Float32, n_events-1);
	} else {
	    tfirst = NA_GET1 (time, Float64, 0);
	    tlast = NA_GET1 (time, Float64, n_events-1);
	}

	if (t1 < tfirst || t0 > tlast) {
	    char errmess[SZ_ERRMESS+1];
	    sprintf (errmess,
		"(%.6g, %.6g) does not overlap the time array", t0, t1);
	    PyErr_SetString (PyExc_RuntimeError, errmess);
	    return NULL;
	}

	if (time_type == tFloat32) {
	    i0 = search (time, n_events, (float)t0);
	    i1 = search (time, n_events, (float)t1);
	} else {
	    i0 = search_d (time, n_events, t0);
	    i1 = search_d (time, n_events, t1);
	}

	indices = Py_BuildValue ("(i,i)", i0, i1);

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

static int search (PyArrayObject *time, int n_events, float t) {

	int low, high;		/* current limits of search range */
	int mid;		/* middle of search range */
	float t_mid;		/* time at mid */

	low = 0;
	high = n_events - 1;

	if (t <= NA_GET1 (time, Float32, 0))
	    return (0);

	if (t >= NA_GET1 (time, Float32, high))
	    return (high + 1);

	while (high - low > 1) {

	    mid = (low + high) / 2;
	    t_mid = NA_GET1 (time, Float32, mid);

	    if (t <= t_mid)
		high = mid;
	    else
		low = mid;
	}

	return (high);
}

/* This is a double precision version of search. */

static int search_d (PyArrayObject *time, int n_events, double t) {

	int low, high;
	int mid;
	double t_mid;

	low = 0;
	high = n_events - 1;

	if (t <= NA_GET1 (time, Float64, 0))
	    return (0);

	if (t >= NA_GET1 (time, Float64, high))
	    return (high + 1);

	while (high - low > 1) {

	    mid = (low + high) / 2;
	    t_mid = NA_GET1 (time, Float64, mid);

	    if (t <= t_mid)
		high = mid;
	    else
		low = mid;
	}

	return (high);
}

/* calling sequence for unbinaccum:

   unbinaccum (image, x, y)

    image      i: a 2-D array (Int32, Int16, UInt16, or Float32)
    x, y      io: the arrays of pixel coordinates (Float32)

   ccos_unbinaccum calls unbinImage, which converts an image array ('image')
   to a pseudo time-tag list.  No time array will be created, just x and y
   pixel coordinates.  image is expected to contain integer values.  For
   each pixel of image that contains a positive value, that number of
   elements will be assigned in the x and y arrays, with their values
   being all the same, the coordinates of the current pixel in image.
*/

static PyObject *ccos_unbinaccum (PyObject *self, PyObject *args) {

	PyObject *oimage, *ox, *oy;
	PyArrayObject *image, *x, *y;
	int status;

	if (!PyArg_ParseTuple (args, "OOO", &oimage, &ox, &oy)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	image = NA_InputArray (oimage, tAny, 0);
	x = NA_IoArray (ox, tFloat32, 0);
	y = NA_IoArray (oy, tFloat32, 0);

	status = unbinImage (image, x, y);

	Py_DECREF (image);
	Py_DECREF (x);
	Py_DECREF (y);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_unbinaccum. */

static int unbinImage (PyArrayObject *image,
		PyArrayObject *x, PyArrayObject *y) {

	float im_data_f32;		/* value before finding nearest int */
	int image_type;			/* data type code for image */
	int nx, ny;			/* size of image */
	int n_events;			/* size of x or y array */
	int i, j, k;
	float ix, jy;			/* i and j converted to float */
	int counts;			/* value of image at a pixel */
	int n;				/* loop index over counts */

	image_type = image->descr->type_num;

	/* Check that the data type of the input image is supported. */
	if (image_type != tInt32 && image_type != tInt16 &&
	    image_type != tUInt16 && image_type != tFloat32) {
	    PyErr_SetString (PyExc_RuntimeError,
			"data type of image is not supported");
	    return 1;
	}

	n_events = x->dimensions[0];
	if (y->dimensions[0] < n_events)
	    n_events = y->dimensions[0];

	nx = image->dimensions[1];
	ny = image->dimensions[0];

	/* Now extract counts into arrays. */
	k = 0;

	for (j = 0;  j < ny;  j++) {
	    for (i = 0;  i < nx;  i++) {

		if (image_type == tInt32) {
		    counts = NA_GET2 (image, Int32, j, i);
		} else if (image_type == tInt16) {
		    counts = NA_GET2 (image, Int16, j, i);
		} else if (image_type == tUInt16) {
		    counts = NA_GET2 (image, UInt16, j, i);
		} else {		/* image_type == tFloat32 */
		    im_data_f32 = NA_GET2 (image, Float32, j, i);
		    counts = NINT (im_data_f32);
		}

		if (k+counts > n_events) {
		    PyErr_SetString (PyExc_RuntimeError,
				"x and y arrays are too short");
		    return 1;
		}

		ix = (float)i;
		jy = (float)j;
		for (n = 0;  n < counts;  n++) {
		    /* these coordinates are zero indexed */
		    NA_SET1 (x, Float32, k, ix);
		    NA_SET1 (y, Float32, k, jy);
		    k++;
		}
	    }
	}

	return 0;
}

/* calling sequence for addrandom:

   newseed = addrandom (x, seed, use_clock)

    x         io: an array of pixel coordinates, either X or Y axis (Float32)
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
   equal to -0.5 to +0.5, depending on roundoff.
*/

static PyObject *ccos_addrandom (PyObject *self, PyObject *args) {

	PyObject *ox;
	PyArrayObject *x;
	PyObject *newseed;
	int seed;
	int use_clock;

	if (!PyArg_ParseTuple (args, "Oii", &ox, &seed, &use_clock)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	x = NA_IoArray (ox, tFloat32, 0);

	newseed = addRN (x, seed, use_clock);

	Py_DECREF (x);

	return newseed;
}

/* This is called by ccos_addrandom. */

static PyObject *addRN (PyArrayObject *x, int seed, int use_clock) {

	double normalize;
	int n_events;		/* size of x (number of events) */
	int k;
	float c_x;

	n_events = x->dimensions[0];

	/* Use the system clock to get a seed? */
	if (use_clock)
	    seed = time (NULL);

	/* Dividing by this normalization factor will make the pseudo-random
	   numbers cover the range from -0.5 to +0.5.
	*/
	normalize = 2. * (double)INT_MAX;

	for (k = 0;  k < n_events;  k++) {
	    seed *= RNG_MULTIPLIER;
	    c_x = NA_GET1 (x, Float32, k);
	    c_x += ((double)seed / normalize);
	    NA_SET1 (x, Float32, k, c_x);
	}

	return Py_BuildValue ("i", seed);	/* return newseed */
}

/* calling sequence for convolve1d:

   convolve1d (flat, dopp, axis)

    flat      io: 2-D flat field image array (Float32)
    dopp       i: 1-D array with which flat will be convolved (Float32)
    axis       i: the axis (0 or 1) along which flat will be convolved (int);
                  axis 1 is the more rapidly varying axis

   The middle element of dopp corresponds to no shift.  For example,
   the following dopp would result in no change to flat:
        dopp = array ((0, 0, 0, 1, 0, 0, 0), type=Float32)

   ccos_convolve1d calls convolveWithDopp, which convolves a flat field
   in-place with dopp.  The dopp array is 1-D, while the flat field is 2-D.
   The convolution will be done along just one axis, specified by axis.
*/

static PyObject *ccos_convolve1d (PyObject *self, PyObject *args) {

	PyObject *oflat, *odopp;
	PyArrayObject *flat, *dopp;
	int axis;
	int status;

	if (!PyArg_ParseTuple (args, "OOi", &oflat, &odopp, &axis)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	flat = NA_IoArray (oflat, tFloat32, 0);
	dopp = NA_InputArray (odopp, tFloat32, 0);

	status = convolveWithDopp (flat, dopp, axis);

	Py_DECREF (flat);
	Py_DECREF (dopp);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_convolve1d. */

static int convolveWithDopp (PyArrayObject *flat, PyArrayObject *dopp,
			int axis) {

	int lendopp;		/* size of dopp */
	int m;			/* middle of dopp */
	int nx, ny;		/* size of flat */
	int i, j, k;
	float sum;
	float *c_flat;		/* a copy of one line or column of flat */
	float *c_dopp;		/* a copy of dopp */

	if (flat->nd > 2) {
	    PyErr_SetString (PyExc_RuntimeError, "flat must be only 2-D");
	    return 1;
	}
	if (dopp->nd > 1) {
	    PyErr_SetString (PyExc_RuntimeError, "dopp must be only 1-D");
	    return 1;
	}

	lendopp = dopp->dimensions[0];
	/* nx is the length of the more rapidly varying axis */
	nx = flat->dimensions[1];
	ny = flat->dimensions[0];

	c_dopp = PyMem_Malloc (lendopp * sizeof (float));
	if (axis == 1)
	    c_flat = PyMem_Malloc ((nx + lendopp) * sizeof (float));
	else
	    c_flat = PyMem_Malloc ((ny + lendopp) * sizeof (float));
	if (c_dopp == NULL || c_flat == NULL) {
	    PyErr_NoMemory();
	    return 1;
	}

	/* Copy dopp to scratch. */
	for (k = 0;  k < lendopp;  k++)
	    c_dopp[k] = NA_GET1 (dopp, Float32, k);

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
		    c_flat[m+i] = NA_GET2 (flat, Float32, j, i);

		for (i = 0;  i < nx;  i++) {
		    sum = 0.;
		    for (k = 0;  k < lendopp;  k++)
			sum += (c_dopp[lendopp-1-k] * c_flat[i+k]);
		    NA_SET2 (flat, Float32, j, i, sum);
		}
	    }

	} else {

	    /* axis = 0:  convolve along Y, the less rapidly varying axis. */

	    for (j = 0;  j < ny+lendopp;  j++)
		c_flat[j] = 1.;

	    for (i = 0;  i < nx;  i++) {	/* for each image column */

		for (j = 0;  j < ny;  j++)
		    c_flat[m+j] = NA_GET2 (flat, Float32, j, i);

		for (j = 0;  j < ny;  j++) {
		    sum = 0.;
		    for (k = 0;  k < lendopp;  k++)
			sum += (c_dopp[lendopp-1-k] * c_flat[j+k]);
		    NA_SET2 (flat, Float32, j, i, sum);
		}
	    }
	}

	PyMem_Free (c_flat);
	PyMem_Free (c_dopp);

	return 0;
}

/* calling sequence for extractband:

   extractband (indata, axis, slope, intercept, outdata)

    indata     i: 2-D image array, from which a band will be extracted
                  (either Float32 or Int16)
    axis       i: the axis (0 or 1) along which the band will be extracted (int)
                  axis 1 is the more rapidly varying axis
    slope      i: the slope of the band (pixels per pixel, double)
    intercept  i: the zero point of the band (pixel number, double)
    outdata   io: a 2-D array, into which the extracted data will be put
                  (either Float32 or Int16)

   While indata and outdata may be either Float32 or Int16, they both
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

static PyObject *ccos_extractband (PyObject *self, PyObject *args) {

	PyObject *oindata, *ooutdata;
	int axis;
	double slope, intercept;
	PyArrayObject *indata, *outdata;
	int status;

	if (!PyArg_ParseTuple (args, "OiddO",
			&oindata, &axis, &slope, &intercept, &ooutdata)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	indata = NA_InputArray (oindata, tAny, 0);
	outdata = NA_IoArray (ooutdata, tAny, 0);

	status = extract2DBand (indata, axis, slope, intercept, outdata);

	Py_DECREF (indata);
	Py_DECREF (outdata);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_extractband. */

static int extract2DBand (PyArrayObject *indata,
		int axis, double slope, double intercept,
		PyArrayObject *outdata) {

	int data_type;		/* data type code for indata and outdata */

	int length;		/* shape of outdata = (extr_height, length) */
	int width;		/* size of indata in cross-disp. direction */
	int extr_height;
	int half_height;	/* half of extr_height, fraction truncated */
	double y, y0;
	int i, j, k;		/* loop indices */
	int bounds_error;	/* true if band would be out of bounds */
	/* one element from indata */
	float c_f32;
	short c_i16;

	data_type = indata->descr->type_num;
	if (data_type != outdata->descr->type_num) {
	    PyErr_SetString (PyExc_RuntimeError,
			"indata and outdata must be of the same data type");
	    return 1;
	}

	/* Check that the data type of the input image is supported. */
	if (data_type != tInt16 && data_type != tFloat32) {
	    PyErr_SetString (PyExc_RuntimeError,
			"data type of arrays is not supported");
	    return 1;
	}

	if (axis < 0 || axis > 1) {
	    PyErr_SetString (PyExc_RuntimeError, "axis must be 0 or 1");
	    return 1;
	}

	extr_height = outdata->dimensions[0];	/* expected to be odd */
	half_height = extr_height / 2;		/* truncate */

	/* length is the axis length in the dispersion direction;
	   width is the axis length in the cross-dispersion direction.
	*/
	length = indata->dimensions[axis];
	width = indata->dimensions[1-axis];

	if (length != outdata->dimensions[1]) {
	    PyErr_SetString (PyExc_RuntimeError,
		"second axis of outdata must agree with size of indata");
	    return 1;
	}

	/* Check for out of bounds. */
	bounds_error = 0;
	y0 = intercept - half_height;
	j = NINT (y0);
	if (j < 0)
	    bounds_error = 1;
	y0 = intercept - half_height + (length-1) * slope;
	j = NINT (y0);
	if (j < 0)
	    bounds_error = 1;

	y0 = intercept + half_height;
	j = NINT (y0);
	if (j >= width)
	    bounds_error = 1;
	y0 = intercept + half_height + (length-1) * slope;
	j = NINT (y0);
	if (j >= width)
	    bounds_error = 1;

	if (bounds_error) {
	    PyErr_SetString (PyExc_RuntimeError,
		"the band would extend beyond the boundary of indata");
	    return 1;
	}

	if (axis == 1) {			/* dispaxis = 1 */

	    for (k = 0;  k < extr_height;  k++) {
		y0 = k + (intercept - half_height);
		for (i = 0;  i < length;  i++) {
		    y = y0 + slope * i;
		    j = NINT (y);
		    /* output[k,i] = input[j,i] */
		    if (data_type == tInt16) {
			c_i16 = NA_GET2 (indata, Int16, j, i);
			NA_SET2 (outdata, Int16, k, i, c_i16);
		    } else {
			c_f32 = NA_GET2 (indata, Float32, j, i);
			NA_SET2 (outdata, Float32, k, i, c_f32);
		    }
		}
	    }

	} else {				/* axis = 0, dispaxis = 2 */

	    for (k = 0;  k < extr_height;  k++) {
		y0 = k + (intercept - half_height);
		for (i = 0;  i < length;  i++) {
		    y = y0 + slope * i;
		    j = NINT (y);
		    /* output[k,i] = input[i,j] */
		    if (data_type == tInt16) {
			c_i16 = NA_GET2 (indata, Int16, i, j);
			NA_SET2 (outdata, Int16, k, i, c_i16);
		    } else {
			c_f32 = NA_GET2 (indata, Float32, i, j);
			/*                                ^^^^        */
			/* swapped, as compared with previous section */
			NA_SET2 (outdata, Float32, k, i, c_f32);
		    }
		}
	    }
	}

	return (0);
}

/* calling sequence for smoothbkg:

   smoothbkg (data, width)

    data      io: a 1-D array to be smoothed in-place (Float32)
    width      i: the width (pixels) of the boxcar smoothing function (int)

   ccos_smoothbkg calls smoothBackground, which boxcar smooths the
   1-D array 'data' in-place, with a width of 'width' pixels.
*/

static PyObject *ccos_smoothbkg (PyObject *self, PyObject *args) {

	PyObject *odata;
	PyArrayObject *data;
	int width;
	int status;

	if (!PyArg_ParseTuple (args, "Oi", &odata, &width)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	data = NA_IoArray (odata, tFloat32, 0);

	status = smoothBackground (data, width);

	Py_DECREF (data);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_smoothbkg. */

static int smoothBackground (PyArrayObject *data, int width) {

	int length;		/* length of data */
	double sum;
	float *scr;		/* temporary copy of data, extended at ends */
	int offset;		/* width / 2, truncated */
	int i, ilow, ihigh;
	float c_data;		/* one element of data */

	length = data->dimensions[0];

	if ((scr = PyMem_Malloc (length+width * sizeof (float))) == NULL) {
	    PyErr_NoMemory();
	    return 1;
	}
	memset (scr, 0, (length+width) * sizeof (float));

	offset = width / 2;

	/* copy to scratch */
	for (i = 0;  i < length;  i++)
	    scr[i+offset] = NA_GET1 (data, Float32, i);

	/* duplicate the leftmost and rightmost elements of data in scratch */
	c_data = NA_GET1 (data, Float32, 0);
	for (i = 0;  i < offset;  i++)
	    scr[i] = c_data;
	c_data = NA_GET1 (data, Float32, length-1);
	for (i = 0;  i < offset;  i++)
	    scr[i+length+offset] = c_data;

	sum = 0.;
	for (i = 0;  i < width-1;  i++)
	    sum += scr[i];

	for (i = offset;  i < length+offset;  i++) {
	    ilow = i - offset - 1;
	    ihigh = ilow + width;
	    sum += scr[ihigh];
	    if (ilow >= 0)
		sum -= scr[ilow];
	    c_data = sum / width;
	    NA_SET1 (data, Float32, i-offset, c_data);
	}

	PyMem_Free (scr);

	return (0);
}

/* calling sequence for addlines:

   addlines (intensity, wavelength, reswidth, x1d_wl, template)

    intensity   i: array of amplitudes of lines (Float32)
    wavelength  i: array of wavelengths of lines (Float64)
    reswidth    i: FWHM of Gaussian line shape (double)
    x1d_wl      i: wavelength array for template spectrum (Float32 or Float64)
    template   io: 1-D template spectrum (Float32)

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

static PyObject *ccos_addlines (PyObject *self, PyObject *args) {

	PyObject *ointensity, *owavelength, *ox1d_wl, *otemplate;
	double reswidth;
	PyArrayObject *intensity, *wavelength, *x1d_wl, *template;
	int status;

	if (!PyArg_ParseTuple (args, "OOdOO",
		&ointensity, &owavelength, &reswidth, &ox1d_wl, &otemplate)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	intensity = NA_InputArray (ointensity, tFloat32, 0);
	wavelength = NA_InputArray (owavelength, tFloat64, 0);
	x1d_wl = NA_InputArray (ox1d_wl, tAny, 0);
	template = NA_IoArray (otemplate, tFloat32, 0);

	status = addEmissionLines (intensity, wavelength,
			reswidth, x1d_wl, template);

	Py_DECREF (intensity);
	Py_DECREF (wavelength);
	Py_DECREF (x1d_wl);
	Py_DECREF (template);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_addlines. */

static int addEmissionLines (PyArrayObject *intensity,
		PyArrayObject *wavelength,
		double reswidth,
		PyArrayObject *x1d_wl, PyArrayObject *template) {

	double *temp_x1d_wl;	/* copy of x1d_wl */
	float *temp_template;	/* copy of template */

	int x1d_wl_type;	/* data type code for x1d_wl */
	double wl;		/* one wavelength from the array */
	float ampl;		/* one value of intensity from the array */
	int nrows;		/* length of intensity & wavelength arrays */
	int nelem;		/* length of x1d_wl and template arrays */
	double x;		/* pixel number */
	double minwl, maxwl;	/* min and max wavelengths in template */
	int i;

	x1d_wl_type = x1d_wl->descr->type_num;
	if (x1d_wl_type != tFloat32 && x1d_wl_type != tFloat64) {
	    PyErr_SetString (PyExc_RuntimeError,
			"data type of x1d_wl is not supported");
	    return 1;
	}

	nrows = wavelength->dimensions[0];
	nelem = x1d_wl->dimensions[0];

	if (nrows != intensity->dimensions[0]) {
	    PyErr_SetString (PyExc_RuntimeError,
		"intensity and wavelength arrays are not the same length");
	    return 1;
	}
	if (nelem != template->dimensions[0]) {
	    PyErr_SetString (PyExc_RuntimeError,
		"x1d_wl and template arrays are not the same length");
	    return 1;
	}

	/* Create scratch arrays for the x1d wavelengths and template. */
	if ((temp_x1d_wl = PyMem_Malloc (nelem * sizeof (double))) == NULL ||
	    (temp_template = PyMem_Malloc (nelem * sizeof (float))) == NULL) {
	    PyErr_NoMemory();
	    return 1;
	}
	if (x1d_wl_type == tFloat32) {
	    for (i = 0;  i < nelem;  i++)
		temp_x1d_wl[i] = NA_GET1 (x1d_wl, Float32, i);
	} else {
	    for (i = 0;  i < nelem;  i++)
		temp_x1d_wl[i] = NA_GET1 (x1d_wl, Float64, i);
	}
	memset (temp_template, 0, nelem * sizeof (float));

	minwl = temp_x1d_wl[nelem-1];
	maxwl = temp_x1d_wl[0];
	if (minwl > maxwl) {
	    wl = minwl;
	    minwl = maxwl;
	    maxwl = wl;
	}

	for (i = 0;  i < nrows;  i++) {
	    wl = NA_GET1 (wavelength, Float64, i);
	    if (wl <= minwl || wl >= maxwl)
		continue;
	    ampl = NA_GET1 (intensity, Float32, i);
	    if (ampl <= 0)
		continue;
	    x = findPixelNumber (wl, temp_x1d_wl, nelem);
	    addLSF (reswidth, ampl, x, temp_template, nelem);
	}

	/* Copy the scratch array back into template. */
	for (i = 0;  i < nelem;  i++)
	    NA_SET1 (template, Float32, i, temp_template[i]);

	PyMem_Free (temp_template);
	PyMem_Free (temp_x1d_wl);

	return (0);
}

static double findPixelNumber (double wl, double x1d_wl[], int nelem) {

	double x;		/* pixel coordinate */
	int i;

	i = binarySearch (wl, x1d_wl, nelem);
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

static int binarySearch (double wl, double array[], int nelem) {

	int low, high;		/* range of elements to consider */
	int k;			/* middle element between low and high */

	if (nelem < 2)
	    return 0;

	if (array[0] < array[1]) {		/* array is increasing */

	    /* check for out of range */
	    if (wl < array[0])
		return -1;
	    else if (wl > array[nelem-1])
		return nelem;

	} else {				/* array is decreasing */

	    if (wl > array[0])
		return -1;
	    else if (wl < array[nelem-1])
		return nelem;
	}

	low = 0;
	high = nelem - 1;

	if (array[0] < array[1]) {

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

static void addLSF (double reswidth, float ampl, double x,
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
	ix = (int) floor (x);
	dx = x - (double)ix;

	/* fwhm / 2 = sigma * sqrt (2 * ln (2)) */
	sigma = reswidth / 2.35482;

	for (i = 0;  i < len_temp;  i++) {
	    x2 = (double)(i - mid) - dx;
	    temp[i] = ampl * exp (-(x2*x2) / (2.*sigma*sigma));
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

   geocorrection (x, y, x_image, y_image, interp_flag, \
		x_offset, y_offset, xbin, ybin)

    x, y       io: arrays of pixel coordinates of the events (Float32)
    x_image     i: the 2-D image array of dx values (Float32)
    y_image     i: the 2-D image array of dy values (Float32)
    interp_flag i: indicates the type of interpolation (int):
                   0 --> use nearest neighbor, 1 --> use bilinear interpolation

   optional arguments:
    x_offset    i: offset in the more rapidly varying axis (int)
    y_offset    i: offset in the less rapidly varying axis (int)
    xbin        i: bin factor in the more rapidly varying axis (int)
    ybin        i: bin factor in the less rapidly varying axis (int)
            x_offset and y_offset are the offsets of x_image and y_image
            from the beginning of the detector, to allow them to be
            subarrays.  x_offset & y_offset are the values of the keywords
            ORIGIN_X and ORIGIN_Y respectively.  These are in units of
            unbinned pixels, and they are zero-indexed.
            xbin and ybin are the bin factors of x_image and y_image
            in the X and Y axes respectively, with default values of 1.
            These are the values of the keywords XBIN and YBIN.

   ccos_geocorrection calls geoInterp2D, which applies the geometric (INL)
   correction to the x and y arrays (in-place).  The corrections to x and y
   are given by the values in x_image and y_image respectively, which have
   offsets of x_offset and y_offset from the origin (so the images do not
   have to be full size).  If interp_flag is true, bilinear interpolation
   within x_image and y_image will be used to get the corrections to x and
   y; otherwise, the nearest pixel in x_image and y_image will be used.
*/

static PyObject *ccos_geocorrection (PyObject *self, PyObject *args) {

	PyObject *ox, *oy, *ox_image, *oy_image;
	int interp_flag;
	int x_offset, y_offset;
	int xbin, ybin;
	PyArrayObject *x, *y, *x_image, *y_image;
	int status;

	x_offset = 0;
	y_offset = 0;
	xbin = 1;
	ybin = 1;

	if (!PyArg_ParseTuple (args, "OOOOi|iiii",
			&ox, &oy, &ox_image, &oy_image, &interp_flag,
			&x_offset, &y_offset, &xbin, &ybin)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	x = NA_IoArray (ox, tFloat32, 0);
	y = NA_IoArray (oy, tFloat32, 0);
	x_image = NA_InputArray (ox_image, tFloat32, 0);
	y_image = NA_InputArray (oy_image, tFloat32, 0);

	status = geoInterp2D (x, y, x_image, y_image, interp_flag,
		(float)x_offset, (float)y_offset, (float)xbin, (float)ybin);

	Py_DECREF (x);
	Py_DECREF (y);
	Py_DECREF (x_image);
	Py_DECREF (y_image);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_geocorrection. */

static int geoInterp2D (PyArrayObject *x, PyArrayObject *y,
	PyArrayObject *x_image, PyArrayObject *y_image, int interp_flag,
	float x_offset, float y_offset, float xbin, float ybin) {

	/* individual values */
	float c_x, c_y;
	/* dx and dy are the values interplated from x_image and y_image;
	   they will be subtracted from the x and y columns to correct
	   the geometric distortion.
	*/
	float dx, dy;
	int nx, ny;		/* size of images */
	int k;			/* loop index for events */
	int n_events;		/* number of rows in events table */
	int i, j;		/* indices in 2-D array */
	float ix, jy;		/* pixel coordinates in x_image and y_image */

	n_events = x->dimensions[0];	/* rows in events table */
	nx = x_image->dimensions[1];
	ny = x_image->dimensions[0];
	if (nx != y_image->dimensions[1] || ny != y_image->dimensions[0]) {
	    PyErr_SetString (PyExc_RuntimeError,
		"x_image and y_image are not the same shape");
	    return 1;
	}

	for (k = 0;  k < n_events;  k++) {

	    /* Get the coordinates of the current event. */
	    c_x = NA_GET1 (x, Float32, k);
	    c_y = NA_GET1 (y, Float32, k);
	    /* Adjust for offset and scale of geo image. */
	    ix = (c_x - x_offset) / xbin;
	    jy = (c_y - y_offset) / ybin;

	    if (interp_flag) {

		if (ix <= -0.5 || ix >= nx-0.5 || jy <= -0.5 || jy >= ny-0.5)
		    continue;
		bilinearInterp (ix, jy, x_image, y_image, nx, ny, &dx, &dy);

	    } else {

		i = NINT (ix);
		j = NINT (jy);
		if (i < 0 || i >= nx || j < 0 || j >= ny)
		    continue;
		dx = NA_GET2 (x_image, Float32, j, i);
		dy = NA_GET2 (y_image, Float32, j, i);
	    }

	    /* Update x and y in-place. */
	    NA_SET1 (x, Float32, k, c_x - dx);
	    NA_SET1 (y, Float32, k, c_y - dy);
	}

	return 0;
}

/* This routine does bilinear interpolation at x,y within the x and y
   image arrays.  The interpolated values are returned as dx and dy.
   x is the more rapidly varying axis.  nx and ny give the size of the
   image arrays.
*/

static void bilinearInterp (float x, float y,
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

	*dx = p * r * NA_GET2 (x_image, Float32, j, i) +     /* lower left */
	      q * r * NA_GET2 (x_image, Float32, j, i+1) +   /* lower right */
	      p * s * NA_GET2 (x_image, Float32, j+1, i) +   /* upper left */
	      q * s * NA_GET2 (x_image, Float32, j+1, i+1);  /* upper right */

	*dy = p * r * NA_GET2 (y_image, Float32, j, i) +
	      q * r * NA_GET2 (y_image, Float32, j, i+1) +
	      p * s * NA_GET2 (y_image, Float32, j+1, i) +
	      q * s * NA_GET2 (y_image, Float32, j+1, i+1);
}

/* calling sequence for interp1d:

   interp1d (x_a, y_a, x_b, y_b)

    x_a, y_a   i: input independent and dependent variable arrays
    x_b        i: independent variable array
    y_b       io: interpolated data at each element of x_b (modified inplace)

   ccos_interp1d calls interp1d_r or interp1d_d (float or double,
   respectively) to interpolate within y_a for each element of x_b.
   All arrays must be 1-D.  x_a and y_a must be the same length, and
   x_b and y_b must be the same length (possibly different from the
   length of x_a).  The arrays will be converted internally to double
   if they aren't already.
*/

static PyObject *ccos_interp1d (PyObject *self, PyObject *args) {

	PyObject *ox_a, *oy_a, *ox_b, *oy_b;
	PyArrayObject *x_a, *y_a, *x_b, *y_b;
	int n_a, n_b;
	int status;

	if (!PyArg_ParseTuple (args, "OOOO",
			&ox_a, &oy_a, &ox_b, &oy_b)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	x_a = NA_InputArray (ox_a, tFloat64, C_ARRAY);
	y_a = NA_InputArray (oy_a, tFloat64, C_ARRAY);
	x_b = NA_InputArray (ox_b, tFloat64, C_ARRAY);
	y_b = NA_IoArray (oy_b, tFloat64, C_ARRAY);

	status = interp_check (x_a, y_a, x_b, y_b);
	if (status) {
	    Py_DECREF (x_a); Py_DECREF (y_a); Py_DECREF (x_b); Py_DECREF (y_b);
	    return NULL;
	}

	n_a = x_a->dimensions[0];
	n_b = x_b->dimensions[0];
	status = interp1d (NA_OFFSETDATA (x_a), NA_OFFSETDATA (y_a), n_a,
			   NA_OFFSETDATA (x_b), NA_OFFSETDATA (y_b), n_b);

	Py_DECREF (x_a);
	Py_DECREF (y_a);
	Py_DECREF (x_b);
	Py_DECREF (y_b);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This function compares the shapes of the arrays.  x_a and y_a must be
   the same length, and x_b and y_b must be the same length (which need not
   be the same as the length of x_a).  All arrays must be 1-D.
*/

static int interp_check (PyArrayObject *x_a, PyArrayObject *y_a,
			 PyArrayObject *x_b, PyArrayObject *y_b) {

	if (x_a->dimensions[0] < 1) {
	    PyErr_SetString (PyExc_RuntimeError,
			"no data in input independent variable array");
	    return 1;
	}

	if (!NA_ShapeEqual (x_a, y_a) || !NA_ShapeEqual (x_b, y_b)) {
	    PyErr_SetString (PyExc_RuntimeError,
			"arrays have inconsistent shapes");
	    return 1;
	}
	if (x_a->nd != 1) {
	    PyErr_SetString (PyExc_RuntimeError,
			"arrays must all be 1-D");
	    return 1;
	}

	return 0;
}

/* This function does linear interpolation to assign values to cy_b.
   For values of cx_a that are outside the range of cx_b, the first
   or last value of cy_a will be assigned to cy_b.
*/

static int interp1d (double cx_a[], double cy_a[], int n_a,
		     double cx_b[], double cy_b[], int n_b) {

/* arguments:
double cx_a[]      i: input independent variable array
double cy_a[]      i: input dependent variable array
int n_a            i: size of cx_a and cy_a (at least 1)
double cx_b[]      i: independent variable array
double cy_b[]     io: interpolated value at each element of x_b
int n_b            i: size of cx_b and cy_b
*/

	int i_a;	/* index in cx_a or cy_a */
	int i_b;	/* index in cx_b or cy_b */
	double p, q;	/* weights for linear interpolation */

	if (n_a == 1) {

	    for (i_b = 0;  i_b < n_b;  i_b++)
		cy_b[i_b] = cy_a[0];

	} else {

	    for (i_b = 0;  i_b < n_b;  i_b++) {

		i_a = binarySearch (cx_b[i_b], cx_a, n_a);

		/* extrapolate with first or last value, if out of bounds */
		if (i_a == -1) {
		    cy_b[i_b] = cy_a[0];
		} else if (i_a == n_a) {
		    cy_b[i_b] = cy_a[n_a-1];
		} else {
		    q = (cx_b[i_b] - cx_a[i_a]) / (cx_a[i_a+1] - cx_a[i_a]);
		    p = 1. - q;
		    cy_b[i_b] = p * cy_a[i_a] + q * cy_a[i_a+1];
		}
	    }
	}
	return 0;
}

/* calling sequence for getstartstop:

   getstartstop (time, y, dq, istart, istop, delta_t)

    time          i: array of times (seconds) of the events (Float32)
    y             i: array of Y pixel coordinates of the events
                     (either Float32 or Int16)
    dq            i: array of data quality flags for the events (Int16)
    istart        o: array of indices in events list of the start of
                     time intervals (Int32)
    istop         o: array of indices of the end of time intervals (Int32)
    delta_t       i: length (seconds) of each time interval (double)

   time, y and dq do not need to be C arrays (i.e. they don't have to be
   contiguous, aligned, or in native byte order).  The istart and istop
   arrays are expected to be C arrays, however.

   ccos_getstartstop gets the indices in the events list of the start and
   stop times of the time intervals.  istart[i] and istop[i] are intended
   to be used as the limits of a Python slice for time interval i.
*/

static PyObject *ccos_getstartstop (PyObject *self, PyObject *args) {

	PyObject *otime, *oy, *odq, *oistart, *oistop;
	PyArrayObject *time, *y, *dq, *istart, *istop;
	double delta_t;
	int status;

	int nbins;		/* length of istart and istop arrays */

	if (!PyArg_ParseTuple (args, "OOOOOd",
			&otime, &oy, &odq, &oistart, &oistop, &delta_t)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	time = NA_InputArray (otime, tFloat32, 0);
	y = NA_InputArray (oy, tAny, 0);
	dq = NA_InputArray (odq, tInt16, 0);
	istart = NA_IoArray (oistart, tInt32, C_ARRAY);
	istop = NA_IoArray (oistop, tInt32, C_ARRAY);

	nbins = istart->dimensions[0];

	status = getStartStopTimes (time, y, dq,
		NA_OFFSETDATA (istart), NA_OFFSETDATA (istop),
		nbins, delta_t);

	Py_DECREF (time);
	Py_DECREF (y);
	Py_DECREF (dq);
	Py_DECREF (istart);
	Py_DECREF (istop);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This function, called by ccos_getstartstop, finds the indices in the
   events list of the start and stop times of the time intervals (nbins
   such intervals, uniformly spaced in time).  The number of source and
   background counts within each such interval will also be found.
*/

static int getStartStopTimes (PyArrayObject *time, PyArrayObject *y,
		PyArrayObject *dq,
		int istart[], int istop[],
		int nbins, double delta_t) {

/* arguments:
time			i: time at each event
y			i: y location of each event
dq			i: data quality flag for each event
istart, istop		o: arrays of start and stop event numbers
nbins                   i: length of istart, istop
delta_t			i: length of time interval
*/

	int y_type;		/* data type code */
	int n_events;		/* number of events (size of time, y, dq) */
	int i;			/* index for istart, istop */
	int k;			/* index for events */
	double t0;		/* time of the first event (close to 0) */
	double end_interval;	/* time at end of a delta_t interval */
	/* individual values */
	float c_time;

	y_type = y->descr->type_num;

	/* Check the data type of the array y. */
	if (y_type != tFloat32 && y_type != tInt16) {
	    PyErr_SetString (PyExc_RuntimeError,
			"y must be either Float32 or Int16");
	    return 1;
	}

	n_events = time->dimensions[0];

	/* Fill in the start and stop index of each time interval.
	   The istart and istop values are intended to be used as limits
	   of a Python slice, e.g. time[istart[i]:istop[i]].
	*/
	t0 = NA_GET1 (time, Float32, 0);
	istart[0] = 0;
	end_interval = t0 + delta_t;
	for (k = 0, i = 0;  k < n_events;  k++) {

	    c_time = NA_GET1 (time, Float32, k);
	    if (c_time >= end_interval) {
		istop[i] = k;
		if (i >= nbins - 1) {
		    istop[nbins-1] = n_events;
		    break;
		} else {
		    i++;
		    istart[i] = k;
		    end_interval = t0 + (i+1) * delta_t;
		}
	    }
	}
	istop[nbins-1] = n_events;

	return 0;
}

/* calling sequence for getbkgcounts:

   getbkgcounts (y, dq,
	istart, istop, bkg_counts, src_counts,
	bkg1_low, bkg1_high, bkg2_low, bkg2_high,
	src_low, src_high, bkgsf)

    y             i: array of Y pixel coordinates of the events
                     (either Float32 or Int16)
    dq            i: array of data quality flags for the events (Int16)
    istart        i: array of indices in events list of the start of
                     time intervals (Int32)
    istop         i: array of indices of the end of time intervals (Int32)
    bkg_counts    o: array of background counts in each interval (Int32)
    src_counts    o: array of source counts in each interval (Int32)
    bkg1_low,     i: row numbers of lower background region (int)
      bkg1_high
    bkg2_low,     i: row numbers of upper background region (int)
      bkg2_high
    src_low,      i: row numbers of source region (int)
      src_high
    bkgsf         i: scale factor to estimate how many background counts
                     there are in the source extraction region (double)

   y and dq do not need to be C arrays (i.e. they don't have to be
   contiguous, aligned, or in native byte order).  The other arrays
   are expected to be C arrays, however:
	istart, istop, bkg_counts, src_counts

   ccos_getbkgcounts gets the number of source and background counts within
   each interval [istart[i]:istop[i]].
*/

static PyObject *ccos_getbkgcounts (PyObject *self, PyObject *args) {

	PyObject *oy, *odq, *oistart, *oistop,
		*obkg_counts, *osrc_counts;
	PyArrayObject *y, *dq, *istart, *istop,
		*bkg_counts, *src_counts;
	int bkg1_low, bkg1_high, bkg2_low, bkg2_high;
	int src_low, src_high;
	double bkgsf;
	int status;

	int nbins;		/* length of bkg_counts (and other) arrays */

	if (!PyArg_ParseTuple (args, "OOOOOOiiiiiid",
			&oy, &odq, &oistart, &oistop,
			&obkg_counts, &osrc_counts,
			&bkg1_low, &bkg1_high, &bkg2_low, &bkg2_high,
			&src_low, &src_high, &bkgsf)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	y = NA_InputArray (oy, tAny, 0);
	dq = NA_InputArray (odq, tInt16, 0);
	istart = NA_InputArray (oistart, tInt32, C_ARRAY);
	istop = NA_InputArray (oistop, tInt32, C_ARRAY);
	bkg_counts = NA_IoArray (obkg_counts, tInt32, C_ARRAY);
	src_counts = NA_IoArray (osrc_counts, tInt32, C_ARRAY);

	nbins = bkg_counts->dimensions[0];

	status = getBkgCounts (y, dq,
		NA_OFFSETDATA (istart), NA_OFFSETDATA (istop),
		NA_OFFSETDATA (bkg_counts), NA_OFFSETDATA (src_counts),
		nbins,
		bkg1_low, bkg1_high, bkg2_low, bkg2_high,
		src_low, src_high, bkgsf);

	Py_DECREF (y);
	Py_DECREF (dq);
	Py_DECREF (istart);
	Py_DECREF (istop);
	Py_DECREF (bkg_counts);
	Py_DECREF (src_counts);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This function, called by ccos_getbkgcounts, finds the number of
   source and background counts within each such interval.
*/

static int getBkgCounts (PyArrayObject *y, PyArrayObject *dq,
		int istart[], int istop[],
		int bkg_counts[], int src_counts[],
		int nbins,
		int bkg1_low, int bkg1_high, int bkg2_low, int bkg2_high,
		int src_low, int src_high, double bkgsf) {

/* arguments:
y			i: y location of each event
dq			i: data quality flag for each event
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
	short c_dq;

	y_type = y->descr->type_num;

	/* Check the data type of the array y. */
	if (y_type != tFloat32 && y_type != tInt16) {
	    PyErr_SetString (PyExc_RuntimeError,
			"y must be either Float32 or Int16");
	    return 1;
	}

	/* Fill in the values for the number of source and background
	   counts within each time interval.
	*/
	for (i = 0;  i < nbins;  i++) {
	    n_src = 0;
	    n_bkg = 0;
	    for (k = istart[i];  k < istop[i];  k++) {
		c_dq = NA_GET1 (dq, Int16, k);
		if (c_dq == 0) {
		    if (y_type == tFloat32) {
			c_y = NA_GET1 (y, Float32, k);
			jy = NINT (c_y);
		    } else {
			jy = NA_GET1 (y, Int16, k);
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

   smallerbursts (time, dq,
	istart, istop, bkg_counts, src_counts,
	smallest_burst, stdrej, source_frac,
	half_block, max_iter,
	large_burst, small_burst, dq_burst, verbose)

    time            i: array of times, used only if verbose (Float32)
    dq             io: array of data quality flags for the events (Int16)
    istart          i: array of indices in events list of the start of
                       time intervals (Int32)
    istop           i: array of indices of the end of time intervals (Int32)
    bkg_counts     io: array of background counts in each interval (Int32)
    src_counts      i: array of source counts in each interval (Int32)
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

   time and dq do not need to be C arrays (i.e. they don't have to be
   contiguous, aligned, or in native byte order).  The other arrays
   are expected to be C arrays, however:
	istart, istop, bkg_counts, src_counts

   ccos_smallerbursts screens for smaller bursts.  If any are found,
   they will be flagged by setting elements of the dq array to dq_burst,
   and the bkg_counts element for each burst will be set to small_burst
   (assumed to be negative).
*/

static PyObject *ccos_smallerbursts (PyObject *self, PyObject *args) {

	PyObject *otime, *odq, *oistart, *oistop, *obkg_counts, *osrc_counts;
	PyArrayObject *time, *dq, *istart, *istop, *bkg_counts, *src_counts;
	double delta_t, smallest_burst, stdrej, source_frac;
	int half_block, max_iter,
		large_burst, small_burst, dq_burst, verbose;
	int nbins;		/* length of bkg_counts (and other) arrays */
	int status;

	if (!PyArg_ParseTuple (args, "OOOOOOddddiiiiii",
			&otime, &odq, &oistart, &oistop,
			&obkg_counts, &osrc_counts,
			&delta_t, &smallest_burst, &stdrej, &source_frac,
			&half_block, &max_iter,
			&large_burst, &small_burst, &dq_burst, &verbose)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	time = NA_InputArray (otime, tFloat32, 0);
	dq = NA_IoArray (odq, tInt16, 0);
	istart = NA_InputArray (oistart, tInt32, C_ARRAY);
	istop = NA_InputArray (oistop, tInt32, C_ARRAY);
	bkg_counts = NA_IoArray (obkg_counts, tInt32, C_ARRAY);
	src_counts = NA_InputArray (osrc_counts, tInt32, C_ARRAY);

	nbins = bkg_counts->dimensions[0];

	status = findSmallerBursts (time, dq,
		NA_OFFSETDATA (istart), NA_OFFSETDATA (istop),
		NA_OFFSETDATA (bkg_counts), NA_OFFSETDATA (src_counts),
		nbins, delta_t,
		smallest_burst, stdrej, source_frac,
		half_block, max_iter,
		large_burst, small_burst, dq_burst, verbose);

	Py_DECREF (time);
	Py_DECREF (dq);
	Py_DECREF (istart);
	Py_DECREF (istop);
	Py_DECREF (bkg_counts);
	Py_DECREF (src_counts);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
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
	delta_counts > stdrej * sqrt (bkg_counts[i])
	delta_counts > source_frac * src_counts[i]
*/

static int findSmallerBursts (PyArrayObject *time, PyArrayObject *dq,
	int istart[], int istop[],
	int bkg_counts[], int src_counts[], int nbins, double delta_t,
	double smallest_burst, double stdrej, double source_frac,
	int half_block, int max_iter,
	int large_burst, int small_burst, int dq_burst, int verbose) {

/*
arguments:
time			 i: time at each event
dq			io: data quality flag for each event
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

	m_filtered = malloc (nbins * sizeof (int));
	if (m_filtered == NULL)
	    return 1;

	for (iter = 0;  iter < max_iter;  iter++) {

	    nreject = 0;
	    if (median_boxcar (bkg_counts, m_filtered,
			nbins, half_block, large_burst))
		return 1;

	    for (i = 0;  i < nbins;  i++) {
		delta_counts = bkg_counts[i] - m_filtered[i];
		if (bkg_counts[i] > 0 &&
		    delta_counts > smallest_burst &&
		    delta_counts > stdrej * sqrt ((double)bkg_counts[i]) &&
		    delta_counts > source_frac * src_counts[i]) {

		    nreject++;
		    if (verbose) {
			c_time = NA_GET1 (time, Float32, istart[i]);
			printf ("burst at time %d, counts = %d,"
				" median = %d, diff = %d, source = %d\n",
				(int)(c_time + delta_t/2.),
				bkg_counts[i], m_filtered[i],
				delta_counts, src_counts[i]);
		    }
		    for (k = istart[i];  k <= istop[i];  k++) {
			c_dq = NA_GET1 (dq, Int16, k);
			c_dq |= dq_burst;
			NA_SET1 (dq, Int16, k, c_dq);
		    }
		    bkg_counts[i] = small_burst;
		}
	    }
	    if (verbose) {
		if (nreject < 1) {
		    if (iter == 0)
			printf ("No small burst detected.\n");
		    else
			printf ("No further bursts detected after"
				" iteration %d.\n", iter+1);
		} else {
		    printf ("After iteration %d, we found %d intervals"
			" affected by bursts.\n", iter+1, nreject);
		}
	    }
	    if (nreject < 1)
		break;
	}
	free (m_filtered);

	return 0;
}

/* Boxcar smooth bkg_counts, with a box of width (2 * half_block + 1).
   Within the box, the smoothed value is the median of all non-negative
   values (previously detected bursts will have been flagged by replacing
   the value in bkg_counts with a negative number).  As an endpoint is
   approached, the box will be truncated on the side of the endpoint.

   The result is returned in the array m_filtered.
*/

static int median_boxcar (int bkg_counts[], int m_filtered[],
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

	temp = malloc ((2*half_block+1) * sizeof (int));
	if (temp == NULL)
	    return 1;

	for (i = 0;  i < nbins;  i++) {
	    /* Shrink the filter near the endpoints. */
	    i0 = i - half_block;
	    if (i0 < 0) i0 = 0;
	    i1 = i + half_block;
	    if (i1 >= nbins) i1 = nbins-1;
	    lenblk = i1 - i0 + 1;

	    memcpy (temp, bkg_counts+i0, lenblk*sizeof (int));
	    qsort (temp, lenblk, sizeof (int), compare_int);
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
	free (temp);
	return 0;
}

static int compare_int (const void *vp, const void *vq) {

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

   badtime = getbadtime (time, dq)

    time       i: array of times of the events (Float32)
    dq         i: array of data quality flags for the events (Int16)

   The function value is the sum of the time intervals within which
   the data quality flag was non-zero for every event in the intervals.
*/

static PyObject *ccos_getbadtime (PyObject *self, PyObject *args) {

	PyObject *otime, *odq;
	PyArrayObject *time, *dq;
	PyObject *badtime;

	if (!PyArg_ParseTuple (args, "OO", &otime, &odq)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	time = NA_InputArray (otime, tFloat32, 0);
	dq = NA_InputArray (odq, tInt16, 0);

	badtime = getBadTime (time, dq);

	Py_DECREF (time);
	Py_DECREF (dq);

	return badtime;
}

static PyObject *getBadTime (PyArrayObject *time, PyArrayObject *dq) {

	double badtime;		/* sum of bad time intervals */
	int in_bad_interval;	/* true if we're in a flagged interval */
	int k;
	double c_t0, c_t1;	/* times at limits of a bad time interval */
	short c_dq;		/* one element from data quality array */

	badtime = 0.;
	in_bad_interval = 0;
	c_t0 = 0.;		/* initialization shouldn't be necessary */
	for (k = 0;  k < time->dimensions[0];  k++) {
	    c_dq = NA_GET1 (dq, Int16, k);
	    if (c_dq != 0) {
		if (!in_bad_interval) {
		    in_bad_interval = 1;
		    c_t0 = NA_GET1 (time, Float32, k);
		}
	    } else if (in_bad_interval) {
		in_bad_interval = 0;			/* dq[k] = 0 */
		/* end of interval is previous time */
		c_t1 = NA_GET1 (time, Float32, k-1);
		badtime += (c_t1 - c_t0);
	    }
	}

	return Py_BuildValue ("d", badtime);	/* return badtime */
}

/* calling sequence for xy_extract:

   xy_extract (xi, eta, dq, extr_height, slope, intercept, spectrum)

    xi, eta     i: arrays of pixel coordinates of the events
                   (either Float32 or Int16); xi is in the dispersion
                   direction, eta is cross-dispersion
    dq          i: array of data qualify flags (Int16)
    extr_height i: cross-dispersion width of the spectral extraction region
    slope       i: the slope of the band (pixels per pixel, double)
    intercept   i: the zero point of the band (pixel number, double)
    spectrum   io: a 1-D array, into which the extracted spectrum will be put
                   (Float64)

   xi and eta may be either Float32 or Int16, and they do not need to be
   the same type.

   slope is the slope of the band with respect to the axis along which
   it will be extracted.  If slope=+0.1, for example, then the band is
   about six degrees counterclockwise from the horizontal axis (or
   clockwise from the vertical axis for NUV).

   ccos_xy_extract calls extrFromEvents, which first initializes the output
   spectrum to zero and then increments elements of the spectrum, one for
   each event within the spectral extraction region.  The output spectrum
   will be 1-D.

   The location of the region to be extracted is specified by slope and
   intercept.  The slope is in pixels per pixel.  The intercept is zero
   indexed, at the eta = 0 edge; this is the location where the center line
   of the region to be extracted meets the edge.  No interpolation is done.
*/

static PyObject *ccos_xy_extract (PyObject *self, PyObject *args) {

	PyObject *oxi, *oeta, *odq, *ospectrum;
	PyArrayObject *xi, *eta, *dq, *spectrum;
	int extr_height;
	double slope, intercept;
	int length;
	int status;

	if (!PyArg_ParseTuple (args, "OOOiddO",
			&oxi, &oeta, &odq, &extr_height, &slope, &intercept,
			&ospectrum)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	xi = NA_InputArray (oxi, tAny, 0);
	eta = NA_InputArray (oeta, tAny, 0);
	dq = NA_InputArray (odq, tInt16, 0);
	spectrum = NA_IoArray (ospectrum, tFloat64, C_ARRAY);

	length = spectrum->dimensions[0];
	status = extrFromEvents (xi, eta, dq, extr_height, slope, intercept,
			NA_OFFSETDATA (spectrum), length);

	Py_DECREF (xi);
	Py_DECREF (eta);
	Py_DECREF (dq);
	Py_DECREF (spectrum);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_xy_extract. */

static int extrFromEvents (PyArrayObject *xi, PyArrayObject *eta,
		PyArrayObject *dq,
		int extr_height, double slope, double intercept,
		double spectrum[], int length) {

	int xi_type, eta_type;	/* data type code for xi and eta */
	int n_events;		/* length of xi and eta arrays */
	int half_height;	/* half of extr_height, fraction truncated */
	/* lower is the lower limit of the spectral extraction region for a
	   given value of xi.
	*/
	double lower;
	double y0;		/* lower edge of spec extr region at xi=0 */
	/* left is the lower limit of the spectral extraction region for
	   a given value of eta.
	*/
	int k;			/* event index */
	/* xi and eta for one event */
	float c_xi, c_eta;
	int c_dq;
	int i, j;		/* nearest integers to c_xi, c_eta */

	n_events = xi->dimensions[0];
	if (n_events != eta->dimensions[0]) {
	    PyErr_SetString (PyExc_RuntimeError,
			"xi and eta must both be the same length");
	    return 1;
	}

	xi_type = xi->descr->type_num;
	eta_type = eta->descr->type_num;

	/* Check that the data types of xi and eta are supported. */
	if ((xi_type != tInt16 && xi_type != tFloat32) ||
	    (eta_type != tInt16 && eta_type != tFloat32)) {
	    PyErr_SetString (PyExc_RuntimeError,
			"xi and eta must be either Int16 or Float32");
	    return 1;
	}

	half_height = extr_height / 2;		/* truncate */

	for (i = 0;  i < length;  i++)
	    spectrum[i] = 0.;

	y0 = intercept - half_height;
	for (k = 0;  k < n_events;  k++) {	/* for each event ... */
	    c_dq = NA_GET1 (dq, Int16, k);
	    if (c_dq == 0) {
		if (xi_type == tInt16) {
		    i = NA_GET1 (xi, Int16, k);
		    c_xi = (double)i;
		} else {
		    c_xi = NA_GET1 (xi, Float32, k);
		    i = NINT (c_xi);
		}
		if (i < 0 || i > length-1)
		    continue;
		if (eta_type == tInt16) {
		    j = NA_GET1 (eta, Int16, k);
		    c_eta = (double)j;
		} else {
		    c_eta = NA_GET1 (eta, Float32, k);
		}
		/* include this event if it's within the spectral region */
		lower = y0 + slope * c_xi;
		if (c_eta >= lower && c_eta <= lower+extr_height)
		    spectrum[i] += 1.;
	    }
	}

	return (0);
}

/* calling sequence for xy_collapse:

   xy_collapse (xi, eta, dq, slope, xdisp)

    xi, eta     i: arrays of pixel coordinates of the events
                   (either Float32 or Int16); xi is in the dispersion
                   direction, eta is cross-dispersion
    dq          i: array of data qualify flags (Int16)
    slope       i: the slope of the band (pixels per pixel, double)
    xdisp      io: a 1-D array, into which the collapsed data will be put;
                   the location of a feature in this array shows where the
                   spectrum crosses the left edge (if FUV) or bottom edge
                   (if NUV) of the detector (Float64)

   xi and eta may be either Float32 or Int16, and they do not need to be
   the same type.

   slope is the slope (in pixels per pixel) of the band with respect to
   the axis along which it will be extracted.  See the description for
   xy_extract.

   ccos_xy_collapse calls collapseFromEvents, which first initializes
   xdisp to zero and collapses the data along the dispersion direction,
   incrementing xdisp for each event.  The length of xdisp should be
   the width of the detector in the cross-dispersion direction.  If the
   slope of the spectrum were zero, element i of xdisp would be incremented
   for each element of eta that is between i-0.5 and i+0.5.  For a non-zero
   slope, each eta position is first adjusted by subtracting (slope * xi),
   so the position is projected to the left edge (if FUV) or bottom edge
   (if NUV).  Note that this is the same convention as for the intercept
   as given in the xtractab reference table.
*/

static PyObject *ccos_xy_collapse (PyObject *self, PyObject *args) {

	PyObject *oxi, *oeta, *odq, *oxdisp;
	PyArrayObject *xi, *eta, *dq, *xdisp;
	double slope;
	int length;
	int status;

	if (!PyArg_ParseTuple (args, "OOOdO",
			&oxi, &oeta, &odq, &slope, &oxdisp)) {
	    PyErr_SetString (PyExc_RuntimeError, "can't read arguments");
	    return NULL;
	}

	xi = NA_InputArray (oxi, tAny, 0);
	eta = NA_InputArray (oeta, tAny, 0);
	dq = NA_InputArray (odq, tInt16, 0);
	xdisp = NA_IoArray (oxdisp, tFloat64, C_ARRAY);

	length = xdisp->dimensions[0];
	status = collapseFromEvents (xi, eta, dq, slope,
			NA_OFFSETDATA (xdisp), length);

	Py_DECREF (xi);
	Py_DECREF (eta);
	Py_DECREF (dq);
	Py_DECREF (xdisp);

	if (status) {
	    return NULL;
	} else {
	    Py_INCREF (Py_None);
	    return Py_None;
	}
}

/* This is called by ccos_xy_collapse. */

static int collapseFromEvents (PyArrayObject *xi, PyArrayObject *eta,
		PyArrayObject *dq,
		double slope, double xdisp[], int length) {

	int xi_type, eta_type;	/* data type code for xi and eta */
	int n_events;		/* length of xi and eta arrays */
	int k;			/* event index */
	/* xi and eta for one event */
	float c_xi, c_eta;
	int c_dq;
	int i, j;		/* nearest integers to c_xi, c_eta */

	n_events = xi->dimensions[0];
	if (n_events != eta->dimensions[0]) {
	    PyErr_SetString (PyExc_RuntimeError,
			"xi and eta must both be the same length");
	    return 1;
	}

	xi_type = xi->descr->type_num;
	eta_type = eta->descr->type_num;

	/* Check that the data types of xi and eta are supported. */
	if ((xi_type != tInt16 && xi_type != tFloat32) ||
	    (eta_type != tInt16 && eta_type != tFloat32)) {
	    PyErr_SetString (PyExc_RuntimeError,
			"xi and eta must be either Int16 or Float32");
	    return 1;
	}

	for (i = 0;  i < length;  i++)
	    xdisp[i] = 0.;

	for (k = 0;  k < n_events;  k++) {
	    c_dq = NA_GET1 (dq, Int16, k);
	    if (c_dq == 0) {
		if (xi_type == tInt16) {
		    i = NA_GET1 (xi, Int16, k);
		    c_xi = (double)i;
		} else {
		    c_xi = NA_GET1 (xi, Float32, k);
		}
		if (eta_type == tInt16) {
		    j = NA_GET1 (eta, Int16, k);
		    c_eta = (double)j;
		} else {
		    c_eta = NA_GET1 (eta, Float32, k);
		}
		/* shift to where spectrum crosses left (or bottom) edge */
		c_eta -= slope * c_xi;
		j = NINT (c_eta);
		if (j >= 0 && j < length)
		    xdisp[j] += 1.;
	    }
	}

	return (0);
}

static PyMethodDef ccos_methods[] = {
	{"binevents", ccos_binevents, METH_VARARGS,
	"bin events table x & y coordinates to an image array"},

	{"bindq", ccos_bindq, METH_VARARGS,
	"flag regions in a 2-D array according to a DQI table"},

	{"applydq", ccos_applydq, METH_VARARGS,
	"assign data quality flags from a DQI table into an events table column"},

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

	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initccos (void) {

	PyObject *mod;		/* the module */
	PyObject *dict;		/* the module's dictionary */

	mod = Py_InitModule ("ccos", ccos_methods);
	import_libnumarray();

	/* set the doc string */
	dict = PyModule_GetDict (mod);
	PyDict_SetItemString (dict, "__doc__",
		PyString_FromString (DocString()));
}
