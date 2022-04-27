# CalCOS Unit tests
#

The purpose of this directory is for generating unit tests for CalCOS and this is done by using "pytest". 
* All python scripts with names that start with <b>"test_"</b> are unit test module for CalCOS modules.
* A calcos module with the name <b>"average.py"</b> would have a unit test module with the name <b>"test_average.py"</b>
* Every test script is created with the assumption that all the dependencies that calcos requires are satisfied.

Below are descriptions of each test module and the steps used to test some of the CalCOS functions.

|      Script name      |    Script type    |
|:---------------------:|:-----------------:|
|    test_cosutil.py    |    Test script    |
|    test_extract.py    |    Test script    |
|    test_airglow.py    |    Test script    |
|    test_average.py    |    Test script    |
|  test_shift_file.py   |    Test script    |
| generate_tempfiles.py | Supporting script |

> Most unit tests follow 3 simple steps 4 in some special cases
>1. Setup: prepare expected values (values the function is supposed to return)
>2. Test: get test values from the target function (values the function actually returned)
>3. Verify: compare expected value against test values
>4. Clean-up: remove any temp files created during the test.

## 1. test_cosutil.py
Unit tests that have the word <b>"print"</b> in their name using the same algorithm.
- open an IO stream
- initialize the message you want to print
- call the function that is being tested and pass the message string to it.
- redirect the output stream towards the function to catch the printed message
- write the value to a variable
- assert the captured message with the original message.

### test_center_of_quadratic():
unit test for center_of_quadratic(coeff, var)
- create a randomized coeff and var arrays
- follow the math to calculate x_min and x_min_sigma aka center of the quadratic
- x_min = -coeff[1] / (2 * coeff[2])
- x_min_sigma = 0.5 * math.sqrt(var1 / coeff[2] ** 2 + var2 * coeff[1] ** 2 / coeff[2] ** 4)
- assert the expected result with the functions return.

### test_precess():
unit test for precess(t, target)
- set a time in MJD
- create a unit vector toward the target
- calculate the expected coordinates
- assert expected with the actual.

### test_err_frequentist():
unit test for err_frequentist(counts)
- create 3 arrays similar to the test in err_gehrels().
- find the poisson confidence interval for each array.
- assert the result with the expected err_lower and err_upper.

### test_err_gehrels():
unit test for err_gehrels(counts)
test ran
- create 3 arrays, one should be random float values, the other two should be the lower (zero) and upper (one) error estimates
- following the math for calculating the upper limit by taking the sqrt of counts + 0.5 and then adding 1 to the result.
- similarly for the lower we add counts + 0.5 and then counts - counts * (1.0 - 1.0 / (9.0 * counts) - 1.0 / (3.0 * np.sqrt(counts))) ** 3
  we will be able to get the lower array.
- finally assert the upper array and the lower array with the results obtained from err_gehrels().

### test_is_product():
- NOTE:
 no test to be done here since we're checking the file if its a product or not
 the return of the function isProduct() is a boolean hence, assert it directly.
###------------------------------------------------------------------------------------------------------------------------------------
 >Note: Generaly unit test functions that end with the word <b>"exception"</b> follow a different way of testing.
> we use <b>pytest.raises("Type of error")</b> where we specify the error type we expect and if that error is returned the function passes.
###-------------------------------------------------------------------------------------------------------------------------------------

### test_guess_aper_from_locn():
unit test for guessAperFromLocn()
- create lists for LPs and aperture positions (2 in this case).
- use the ranges provided to guess which aperture is being used 
1. LP: 1
- (116.0, 135) ---> PSA
- (-163.0, -143.0) ---> BOA
2. LP: 2
- (52.0, 72.0) ---> PSA
- (-227.0, -207.0) ---> BOA 
3. LP: 3 and above
- aperture will be none
- assert expected positions with the actual position.


## 2. test_extract.py
### test_get_columns():
Test if the function is returning the right column fields

### test_remove_unwanted_column():
Old column length should be equal to new column length + amount of the removed columns

### test_next_power_of_two():
check if function returns the next_power_of_two

### test_add_column_comment():


## 3. test_airglow.py
### test_find_airglow_limits():
unit test for find_airglow_limits()
test ran
- By providing certain values as dict to be used as filter for finding the dispersion
- testing for both FUV segments
- creating a temporary disptab ref file.
- testing for 5 airglow lines
- calculating the expected pixel numbers by following the math involved in the actual file
    and referring to the values in the ref file we can get the values upto a descent decimal points.

## 4. test_average.py
### test_avg_image():
tests avg_image() in average.py
explanation of the test
- create temporary count files to be used as inputs
- expected values in the output file are the average of the input values
- loop though the values to check if the math holds.

## 5. test_shift_file.py
### test_shift_file():
Creates a temporary txt file with some arbitrary values and verify if the shift_file objects are created with the neccessary variables initialized with the right values.

### test_get_shifts():
Instantiate 2 objects with different dataset name and fpoffset and also create a key to use as a filter
the function getShifts() returns a tuple so using a loop find the shift for different combinations of key and finally assert it with the expected values list.

# Supporting script
## 1. generate_tempfiles.py

### create_count_file(file=None):
Creates a temp count file for testing avg_image.

    Parameters
    ----------
    file: str
        the filename string

    Returns
    -------
    filename string

### create_disptab_file(file=None):
Creates a disptab file.

    Parameters
    ----------
    file: str
        name of the temp file to be created.

    Returns
    -------
        name of the temp file created.
### generate_fits_file(file):
Creates a corrtag file for testing.

    Parameters
    ----------
    file: str
        the file path.
    Returns
    -------
    the HDU_List



### All files listed in mentioned here are written by <mark>Michael Asfaw</mark> - masfaw@stsci.edu. 
