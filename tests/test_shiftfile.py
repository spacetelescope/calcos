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
            elif i % 6 == 0 or i % 8 == 0:
                file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("ghi456jkl", "any", "2", "FUVA", "19.543453", "5"))
            elif i == 9:
                file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("mno789pqr", "any", "2", "FUVB", "52.723453", "6"))
            else:
                file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("mno789pqr", "any", "1", "NUVC", "-34.543453", "7"))

    # Test
    ob = shiftfile.ShiftFile(shift_file, 'abc123def', 'any')
    # Verify
    assert len(ob.user_shift_dict) > 0


def test_get_shifts():
    # Setup
    shift_file = "shift_file.txt"
    ob1 = shiftfile.ShiftFile(shift_file, 'ghi456jkl', 'any')
    ob2 = shiftfile.ShiftFile(shift_file, 'abc123def', 'any')
    keys = [('any', 'nuva'), ('any', 'nuvb'), (2, 'nuvc'), ('any', 'any'), ('any', 'fuva'), ('any', 'fuvb')]
    expected_values1 = [((None, None), 0), ((34.543453, 7.0), 1), ((None, None), 0), ((19.543453, 5.0), 2), ((19.543453, 5.0), 1), ((None, None), 0)]
    expected_values2 = [((45.234435, 7.0), 1), ((None, None), 0), ((None, None), 0), ((45.234435, 7.0), 1), ((None, None), 0), ((None, None), 0)]
    # Test
    test_values1 = []
    test_values2 = []
    for key in keys:
        test_values1.append(shiftfile.ShiftFile.getShifts(ob1, key))
    for key in keys:
        test_values2.append(shiftfile.ShiftFile.getShifts(ob2, key))
    # Verify
    for i in range(len(expected_values1)):
        assert expected_values1[i] == test_values1[i]
        assert expected_values2[i] == test_values2[i]
