class ShiftFile (object):
    """Read shift_file.

    The shift_file is a text file supplied by the user to specify either
    shift1 or shift2 (or both).  Blank lines and lines beginning with '#'
    will be ignored; otherwise, every line must have either five or six
    words:

        rootname, fpoffset, flash_number, segment/stripe, shift1, shift2

    All words given as strings are case insensitive (internally they will be
    converted to lower case).
    If shift1 is not to be specified, use the value "N/A".
    If shift2 is not to be specified, it may be given as "N/A" or simply left
    off (i.e. only five words on the line).
    Any or all of rootname, fpoffset, flash_number and segment/stripe may be
    given as "ANY", which is interpreted as a wildcard, i.e. it matches any
    rootname, fpoffset, etc.

    user_shifts = shiftfile.ShiftFile (shift_file, rootname, fpoffset)

    getShifts is a public method:
        ((shift1, shift2), nfound) = user_shifts.getShifts (key)
    key is a tuple of flash number (or "any") and segment/stripe name.

    @ivar shift_file: name of text file supplied by user
    @type shift_file: string
    @ivar rootname: rootname of the current exposure
    @type rootname: string
    @ivar fpoffset: fpoffset of the current exposure
    @type fpoffset: int
    """

    def __init__ (self, shift_file, rootname, fpoffset):

        # This is a dictionary of shifts for the current exposure, with
        # keys (flash_number, segment) and values (shift1, shift2).
        self.user_shift_dict = None

        fd = open (shift_file, "r")
        lines = fd.readlines()
        fd.close()

        user_shift_dict = {}
        for line in lines:
            line = line.strip()
            if line[0] == '#':                  # ignore comments
                continue
            words = line.split()
            if not words:                       # ignore blank lines
                continue
            nwords = len (words)
            if nwords < 5 or nwords > 6:
                raise RuntimeError, \
                "error reading this line of shift_file:  '%s'" % line
            for i in range (nwords):
                words[i] = words[i].lower()
            # Select rows matching rootname and fpoffset.
            if words[0] != "any" and rootname != words[0]:
                continue
            if words[1] != "any" and fpoffset != int (words[1]):
                continue
            if words[2] == "any":
                flash_number = "any"
            else:
                flash_number = int (words[2])
            segment = words[3].lower()          # could be "any"
            key = (flash_number, segment)
            if words[4] != "n/a":
                shift1 = float (words[4])
            else:
                shift1 = None
            if nwords == 6 and words[5] != "n/a":
                shift2 = float (words[5])
            else:
                shift2 = None
            user_shift_dict[key] = (shift1, shift2)

        self.user_shift_dict = user_shift_dict

    def getShifts (self, key):
        """Return the shifts corresponding to key, if any.

        @param key: flash number and segment; if flash number is "any" it
            matches any flash number (use "any" for auto/GO wavecals), and
            if segment is "any" it matches any segment or stripe (strings
            are case insensitive)
        @type key: tuple

        @return: (shift1, shift2) and nfound (the number of elements--which
            should be either 0 or 1--that match key); either shift1 or shift2
            may be None, and they will both be None if nfound is 0
        @rtype: tuple
        """

        (flash_number, segment) = key
        segment = segment.lower()
        if isinstance (flash_number, str):
            flash_number = flash_number.lower()

        nfound = 0
        shifts = (None, None)
        # sf_key is the flash number and segment read from the shift file
        for sf_key in self.user_shift_dict.keys():
            if sf_key[0] == "any" or flash_number == "any" or \
                                     flash_number == sf_key[0]:
                if sf_key[1] == "any" or segment == "any" or \
                                         segment == sf_key[1]:
                    shifts = self.user_shift_dict[sf_key]
                    nfound += 1

        return (shifts, nfound)
