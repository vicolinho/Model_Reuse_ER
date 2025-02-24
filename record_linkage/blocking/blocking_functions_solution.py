""" Module for defining blocking functions. Each function computes for a record and a specified attribute a blocking key
that will be concatenated with blocking keys
"""


# TODO Implement Soundex based blocking
def simple_blocking_key(rec_values, attr):
    """Build the blocking index data_io structure (dictionary) to store blocking
     key values (BKV) as keys and the corresponding list of record identifiers.

     A blocking is implemented that concatenates Soundex encoded values of
     attribute values.

     Parameter Description:
       rec_values      : list of record values
       attr : attribute index

     This method returns a blocking key value for a certain attribute value of a record
  """
    return rec_values[attr]


# TODO Implement Soundex based blocking
def phonetic_blocking_key(rec_values, attr):
    """
     A blocking key is generated using Soundex
     Parameter Description:
       rec_values      : list of record values
       attr : attribute index

     This method returns a blocking key value for a certain attribute value of a record
  """
    rec_bkv = ''
    attr_val = rec_values[attr]
    if len(attr_val) > 0:
        sdx = attr_val[0]  # Keep first letter

        for c in attr_val[1:]:  # Loop over all other letters
            if c in 'aehiouwy':  # Not included into Soundex code
                pass
            elif c in 'bfpv':
                sdx += '1'
            elif c in 'cgjkqsxz':
                sdx += '2'
            elif c in 'dt':
                sdx += '3'
            elif c in 'l':
                sdx += '4'
            elif c in 'mn':
                sdx += '5'
            elif c in 'r':
                sdx += '6'
        # Remove duplicate digits
        #
        sdx2 = sdx[:2]  # Keep initial letter and first digit
        for c in sdx[2:]:
            if (c != sdx2[-1]):
                sdx2 += c

        # Set proper length
        #
        if len(sdx2) > 4:
            sdx3 = sdx2[:4]
        elif len(sdx2) < 4:
            sdx3 = sdx2 + '0' * (4 - len(sdx2))
        else:
            sdx3 = sdx2
        rec_bkv += sdx3
    return rec_bkv


# TODO Implement SLK based blocking
def slk_blocking_key(rec_values, attribute_index_list):
    """Build the blocking index data_io structure (dictionary) to store blocking
     key values (BKV) as keys and the corresponding list of record identifiers.

     A blocking key is generated using statistical linkage key (SLK-581)
     blocking approach as used in real-world linkage applications:

     http://www.aihw.gov.au/WorkArea/DownloadAsset.aspx?id=60129551915

     A SLK-581 blocking key is the based on the concatenation of:
     - 3 letters of family name
     - 2 letters of given name
     - Date of birth
     - Sex

     Parameter Description:
       rec_dict          : Dictionary that holds the record identifiers as
                           keys and corresponding list of record values
       attribute_index_list    : List of attribute indices where each attribute is required for generating SLK-581

     This method returns a blocking key utilizing SLK-581
  """
    # *********** Implement SLK-581 function here ***********
    slk = ''
    ln = rec_values[attribute_index_list[0]]
    if ln == '':  # Last name
        slk += '999'
    else:
        ln = ln.replace('-', '')  # Remove non letter characters
        ln = ln.replace(",", '')
        ln = ln.replace('_', '')

        if len(ln) >= 5:
            slk += ln[1] + ln[2] + ln[4]
        elif len(ln) >= 3:
            slk += ln[1] + ln[2] + '2'
        elif len(ln) >= 2:
            slk += ln[1] + '22'
    fn = rec_values[attribute_index_list[1]]
    if fn == '':  # First name
        slk += '99'
    else:
        fn = fn.replace('-', '')  # Remove non letter characters
        fn = fn.replace(",", '')
        fn = fn.replace('_', '')

        if len(fn) >= 3:
            slk += fn[1] + fn[2]
        elif len(fn) >= 2:
            slk += fn[1] + '2'

    # DoB structure we use: dd/mm/yyyy
    #
    dob = rec_values[attribute_index_list[2]]
    dob_list = dob.split('/')

    # Add some checks
    #
    if len(dob_list[0]) < 2:
        dob_list[0] = '0' + dob_list[0]  # Add leading zero for days < 10
    if len(dob_list[1]) < 2:
        dob_list[1] = '0' + dob_list[1]  # Add leading zero for months < 10

    dob = ''.join(dob_list)  # Create: ddmmyyyy
    assert len(dob) == 8, dob

    slk += dob
    gender = rec_values[attribute_index_list[3]]
    if (gender == 'm'):
        slk += '1'
    elif (gender == 'f'):
        slk += '2'
    else:
        slk += '9'
    slk
    return slk
