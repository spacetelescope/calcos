from calcos import shiftfile


def test_shift_file():  # Tests the initialized variables
    # Setup
    shift_file = "shift_file.txt"

    # Test
    ob = shiftfile.ShiftFile(shift_file, 'lco721egq', 'any')
    # Verify
    assert len(ob.user_shift_dict) > 0


def test_get_shifts():
    # Setup
    shift_file = "shift_file.txt"
    ob = shiftfile.ShiftFile(shift_file, 'lco721egq', 'any')
    keys = [(2, 'any'), (1, 'nuva'), (2, 'nuvc'), (1, 'any')]
    expected_values = [((-41.67250061, 10.0), 3), ((-42.35122681, 10.0), 1),
                       ((-41.67250061, 10.0), 1), ((-41.17250061, 10.0), 3)]
    # todo change the values
    # Test
    test_values = []
    for key in keys:
        test_values.append(shiftfile.ShiftFile.getShifts(ob, key))
    # Verify
    for i in range(len(expected_values)):
        assert expected_values[i] == test_values[i]
