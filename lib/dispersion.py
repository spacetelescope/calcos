import numpy as N
import cosutil

class Dispersion (object):
    """Dispersion relation.

    The public methods are:
        flag = disprel.isValid()
        nrows = disprel.getNRows()
        wavelength = disprel.evalDisp (x)
        dwavelength / dx = disprel.evalDerivDisp (x)
        x = disprel.evalInvDisp (wavelength, tiny=1.e-8)
        disprel.info()
        disprel.close()

    @ivar disptab: name of table containing dispersion relations
    @type disptab: string
    @ivar filter: parameters for selecting a row from the disptab
    @type filter: dictionary
    @ivar use_fpoffset: if True, include fpoffset in the filter; if False,
        exclude it from the filter
    @type use_fpoffset: boolean
    """

    def __init__ (self, disptab, filter, use_fpoffset=True):

        # This will be a local copy of filter, possibly excluding "fpoffset".
        self.filter = {}
        self.use_fpoffset = use_fpoffset
        self.ncoeff = 0
        self.coeff = None
        self.delta = 0.
        self.fpoffset = 0               # save for information
        self._nrows = 0                 # number of matching rows (should be 1)
        self._valid = True

        for key in filter.keys():
            key_lower = key.lower()
            if key_lower == "fpoffset":
                self.fpoffset = filter[key]
                continue
            self.filter[key_lower] = filter[key]
        if use_fpoffset:
            self.filter["fpoffset"] = self.fpoffset

        if not cosutil.findColumn (disptab, "fpoffset"):
            if self.filter.has_key ("fpoffset"):
                del (self.filter["fpoffset"])

        disp_info = cosutil.getTable (disptab, self.filter)
        if disp_info is None:
            self._valid = False
            del disp_info
            return
        else:
            self._nrows = len (disp_info)

        self.ncoeff = disp_info.field ("nelem")[0]
        if self.ncoeff < 2:
            raise ValueError, "Dispersion relation has too few coefficients"
        self.coeff = disp_info.field ("coeff")[0][0:self.ncoeff]
        if cosutil.findColumn (disp_info, "delta"):
            self.delta = disp_info.field ("delta")[0]
        else:
            if cosutil.findColumn (disp_info, "d_tv03"):
                d_tv03 = disp_info.field ("d_tv03")[0]
            else:
                d_tv03 = 0.
            if cosutil.findColumn (disp_info, "d2"):
                d2 = disp_info.field ("d2")[0]
            else:
                d2 = 0.
            # the sign of this difference is uncertain at this time:
            self.delta = d_tv03 - d2            # xxx sign?

        del disp_info

    def info (self):

        print "filter =", self.filter
        print "use_fpoffset =", self.use_fpoffset
        print "fpoffset =", self.fpoffset
        print "number of coefficients =", self.ncoeff
        print "coeff =", self.coeff
        print "delta =", self.delta
        print "number of matching rows =", self._nrows
        print "valid =", self._valid

    def isValid (self):
        """Return True if a matching row was found in disptab."""

        return self._valid

    def getNRows (self):
        """Return the number of rows in disptab that match the filter."""

        return self._nrows

    def close (self):
        """Delete coefficients and reset attributes."""

        del self.coeff
        self.filter = {}
        self.ncoeff = 0
        self.delta = 0.
        self.fpoffset = 0
        self._nrows = 0
        self._valid = False

    def evalDisp (self, x):
        """Evaluate the dispersion relation at x.

        The function value will be the wavelength (or array of wavelengths)
        at x, in Angstroms.

        @param x: pixel coordinate (or array of coordinates)
        @type x: numpy array or float

        @return: wavelength (or array of wavelengths) at x
        @rtype: numpy array or float
        """

        x_prime = x - self.delta

        sum = self.coeff[self.ncoeff-1]
        for i in range (self.ncoeff-2, -1, -1):
            sum = sum * x_prime + self.coeff[i]

        return sum

    def evalDerivDisp (self, x):
        """Evaluate the derivative of the dispersion relation at x.

        The function value will be the slope (or array of slopes) at x,
        in Angstroms per pixel.

        @param x: pixel coordinate (or array of coordinates)
        @type x: numpy array or float

        @return: slope at x, in Angstroms per pixel
        @rtype: numpy array or float
        """

        x_prime = x - self.delta

        sum = (self.ncoeff - 1.) * self.coeff[self.ncoeff-1]
        for n in range (self.ncoeff-2, 0, -1):
            sum = sum * x_prime + n * self.coeff[n]

        return sum

    def evalInvDisp (self, wavelength, tiny=1.e-8):
        """Evaluate the inverse of the dispersion relation at wavelength.

        The function value will be the pixel number (or array of pixel numbers)
        at the specified wavelength(s).  Newton's method is used for finding
        the pixel numbers, and the iterations are stopped when the largest
        difference between the specified wavelengths and computed wavelengths
        is less than tiny.

        @param wavelength: wavelength (or array of wavelengths)
        @type wavelength: numpy array or float
        @param tiny: maximum allowed difference between the final pixel
            number(s) and the value from the previous iteration
        @type tiny: float

        @return: pixel number (or array of pixel numbers) at wavelength
        @rtype: numpy array or float
        """

        tiny = abs (tiny)

        # initial value
        try:
            nelem = len (wavelength)
            x = N.arange (nelem, dtype=N.float64)
        except TypeError:
            nelem = 0
            x = 0.

        # Iterate to find the pixel number(s) x such that evaluating the
        # dispersion relation at that point or points gives the specified
        # wavelength(s).
        done = False
        while not done:
            if nelem > 0:
                x_prev = x.copy()
            else:
                x_prev = x
            wl = self.evalDisp (x)
            slope = self.evalDerivDisp (x)
            wl_diff = wavelength - wl
            x += wl_diff / slope
            diff = N.abs (x - x_prev)
            if diff.max() < tiny:
                done = True

        return x
