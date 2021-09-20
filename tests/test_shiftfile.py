from calcos import shiftfile


def test_shift_file():  # Tests the initialized variables
    # Setup
    shift_file = "shift_file.txt"
    with open(shift_file, "w") as file:
        file.write("#dataset\tfpoffset\tflash #\tstripe\tshift1\tshift2\n")
        for i in range(10):
            if i % 3 == 0:
                file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("abc123def", "any", "1", "NUVA", "45.234435", "7"))
            elif i % 5 == 0:
                file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("ghi456jkl", "any", "2", "NUVB", "34.543453", "7"))
            else:
                file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("mno789pqr", "any", "1", "NUVC", "-34.543453", "7"))

    # Test
    ob = shiftfile.ShiftFile(shift_file, 'abc123def', 'any')
    # Verify
    assert len(ob.user_shift_dict) > 0


def test_get_shifts():
    # Setup
    shift_file = "shift_file.txt"
    ob = shiftfile.ShiftFile(shift_file, 'lco721egq', 'any')
    keys = [(2, 'nuva'), ('any', 'nuvb'), (2, 'nuvc'), (1, 'any')]
    expected_values = [((-41.85122681, 10.0), 1), ((-40.95245361, 10.0), 2),
                       ((-41.67250061, 10.0), 1), ((-41.17250061, 10.0), 3)]
    # Test
    test_values = []
    for key in keys:
        test_values.append(shiftfile.ShiftFile.getShifts(ob, key))
    # Verify
    print(test_values)
    for i in range(len(expected_values)):
        assert expected_values[i] == test_values[i]
