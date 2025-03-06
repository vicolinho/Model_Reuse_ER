""" Module for defining blocking functions. Each function computes for a record and a specified attribute a blocking key
that will be concatenated with blocking keys
"""


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

