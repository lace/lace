# pylint: disable=len-as-condition
def load(f, existing_mesh=None):
    from baiji.serialization.util.openlib import ensure_file_open_and_call
    return ensure_file_open_and_call(f, _load, mode='rb', existing_mesh=existing_mesh)

def _load(f, existing_mesh=None):
    from lace.mesh import Mesh
    parser = BSFParser(f)
    if existing_mesh is None:
        return Mesh(v=parser.v, f=parser.f, vc=parser.vc)
    else:
        existing_mesh.v = parser.v
        existing_mesh.f = parser.f
        existing_mesh.vc = parser.vc
        return existing_mesh


class BSFParser(object):
    def __init__(self, f):
        import numpy as np
        self._parse_header(f.read(256))
        if self.header['second_header_offset'] > 0:
            f.read(self.header['second_header_offset'])
        self.v = []
        self.vc = []
        self.f = []
        self.bsf_column_data = {'imcol':[], 'slice':[], 'cam':[]}
        point = self._parse_point(f)
        while point != None:
            point = self._parse_point(f)
        self.v = (np.array(self.v) - np.array(self.header['offset_xyz'])) * self.header['resolution']
        self.v[:, 2] = -self.v[:, 2]
        self.vc = np.array(self.vc) / 255.0

        # calculate faces
        if len(self.bsf_column_data['imcol']) > 0:
            self.bsf_column_data['imcol'] = np.array(self.bsf_column_data['imcol'])
            self.bsf_column_data['slice'] = np.array(self.bsf_column_data['slice'])
            self.bsf_column_data['cam'] = np.array(self.bsf_column_data['cam'])
            self.bsf_column_data['index'] = np.array(range(len(self.bsf_column_data['cam'])))
            MAX_EDGE_LENGTH = 0.02 # 2cm
            def check_dist(p0, p1):
                return np.linalg.norm(self.v[p0] - self.v[p1]) <= MAX_EDGE_LENGTH
            for cam in set(self.bsf_column_data['cam']):
                inds_for_cam = self.bsf_column_data['cam'] == cam
                slices = sorted(set(self.bsf_column_data['slice'][inds_for_cam]))

                def ordered_inds_and_cols_for_slice(slc):
                    subset = np.logical_and(inds_for_cam, self.bsf_column_data['slice'] == slc) # pylint: disable=cell-var-from-loop
                    order = np.argsort(self.bsf_column_data['imcol'][subset])
                    inds = self.bsf_column_data['index'][subset]
                    inds = inds[order]
                    cols = self.bsf_column_data['imcol'][subset]
                    cols = cols[order]
                    return inds, cols

                for bottom_slice, top_slice in zip(slices[0:-2], slices[1:-1]):
                    top, top_cols = ordered_inds_and_cols_for_slice(top_slice)
                    bottom, bottom_cols = ordered_inds_and_cols_for_slice(bottom_slice)

                    ii_b = 0


                    while ii_b < len(bottom)-1:# and ii_t < len(top):
                        p0 = bottom[ii_b]
                        p1 = bottom[ii_b+1]
                        min_col = bottom_cols[ii_b]
                        ii_b += 1
                        ii_t = 0
                        while ii_t < len(top) and top_cols[ii_t] < min_col:
                            ii_t += 1
                        if ii_t < len(top):
                            p2 = top[ii_t]
                            ii_t += 1
                            if check_dist(p1, p2):
                                if check_dist(p0, p2) and check_dist(p0, p1):
                                    self.f.append([p0, p1, p2])
                                if ii_t < len(top):
                                    p3 = top[ii_t]
                                    ii_t += 1
                                    if check_dist(p1, p3) and check_dist(p2, p3):
                                        self.f.append([p1, p3, p2])
            self.f = np.array(self.f)
            self.f = self.f[:, [0, 2, 1]] # Fix triangle winding for outward pointing normals after z inversion

    @property
    def BSF_SUBTYPES(self):
        from collections import OrderedDict
        return OrderedDict([
            ('RAW_3D', {
                'length': 12,
                'structure': "!6H",
                'v': [0, 1, 2],
                'imcol': 3,
                'slice': 4,
                'cam': 5,
            }),
            ('RAW_3DQ', {
                'length': 14,
                'structure': "!7H",
                'v': [0, 1, 2],
                'imcol': 3,
                'slice': 4,
                'cam': 5,
            }),
            ('COLOR_3D', {
                'unsupported': True
            }),
            ('COLOR_3DQ', {
                'unsupported': True
            }),
            ('BASIC_3D', {
                'length': 6,
                'structure': "!3H",
                'v': [0, 1, 2],
            }),
            ('COMPRESSED_3D', {
                'unsupported': True
            }),
            ('HI_COMPRESSED', {
                'unsupported': True
            }),
            ('UNKNOWN', {
                'length': 26,
                'structure': "!6H14B",
                'v': [0, 1, 2],
                'vc': [6, 6, 6],
                'imcol': 3,
                'slice': 4,
                'cam': 5,
            }),
        ])
    @property
    def _point_info(self):
        info = self.BSF_SUBTYPES[self.header['subtype']]
        if 'unsupported' in info:
            raise ValueError("BSF subtype %s is not supported. Open an issue on github." % self.header['subtype'])
        return info

    def _parse_point(self, f):
        import struct
        MAX_SIGNED = 2**15
        MAX_UNSIGNED = 2**16
        def fix_sign(x):
            if x >= MAX_SIGNED:
                return x - MAX_UNSIGNED
            else:
                return x
        info = self._point_info
        data = f.read(info['length'])
        if data == "" or len(data) != info['length']:
            return None
        data = struct.unpack(info['structure'], data)
        if 'v' in info:
            self.v.append(map(fix_sign, [data[ii] for ii in info['v']]))
        if 'vc' in info:
            self.vc.append([data[ii] for ii in info['vc']])
        if 'imcol' in info:
            self.bsf_column_data['imcol'].append(data[info['imcol']])
        if 'slice' in info:
            self.bsf_column_data['slice'].append(data[info['slice']])
        if 'cam' in info:
            self.bsf_column_data['cam'].append(data[info['cam']])
        return True


    def _parse_header(self, data):
        import six
        import struct
        fields = {
            'identifier': ("3s", 0),
            'offset_xyz': ("3h", 80),
            'production_date': ("16s", 16),
            'producer': ("16s", 128),
            'customer': ("16s", 160),
            'deleted_components': ("i", 176),
            'texture_name': ("16s", 192),
            'subsampling': ("BB", 210),
            'scan_guid': ("16s", 213),
            'counter': ("f", 229),
            'average_quality': ("f", 233),
            'current_quality': ("f", 237),
            'average_scatter_light': ("f", 241),
            'current_scatter_light': ("f", 245),
            'height_calibration_version': ("B", 251),
            'second_header_offset': ("H", 254),
            'scanner': ("16s", 144, {
                '' : 'Pro',
                'PRO_XX' : 'Pro',
                'SMART_XX' : 'Smart',
                'AHEAD_XX' : 'Ahead',
                'PEDUS_XX' : 'Pedus',
                'XX-SMART_XX' : 'XX-Smart',
                'LC-SMART_XX' : 'LC-Smart',
            }),
            'subtype': ("B", 3, self.BSF_SUBTYPES.keys()),
            'purify_strength': ("B", 208, ['PURIFIED_UNKNOWN', 'PURIFIED_NOTHING', 'PURIFIED_NORMAL', 'PURIFIED_STRONG']),
            'smooth_strength': ("B", 209, ['SmoothScanUnknown', 'SmoothScanNothing', 'SmoothScanLittle', 'SmoothScanMore', 'SmoothScanExtreme']),
            'centered': ("B", 212, ['unknown', 'centered', 'not centered']),
            'calibration': ("B", 250, ['Vitronic GK+HK', 'Vitronic GK', 'Vitronic GK + HSHK', 'Vitronic WK (Werkskalibrierung)', 'UNKNOWN']),
            'version': ("B", 4),
            'resolution_mantissa': ("B", 64),
            'resolution_exponent': ("b", 65),
            'arm_checks': ("B", 249)
        }
        self.header = {}
        for field, spec in fields.items():
            val = struct.unpack(spec[0], data[spec[1]:(spec[1] + struct.calcsize(spec[0]))])
            if len(val) == 1:
                val = val[0]
            if isinstance(val, six.string_types):
                val = val.rstrip('\x00')
            if len(spec) > 2:
                try:
                    val = spec[2][val]
                except (IndexError, KeyError):
                    pass
            self.header[field] = val


        if self.header['identifier'] != 'bsf':
            raise NotImplementedError("BSF mesh appears not to be in BSF format.")
        self.header['version'] = "V. 1.%d" % self.header['version']
        self.header['resolution'] = self.header['resolution_mantissa'] * 10**(-self.header['resolution_exponent'])
        ARM_CHECK_VALUES = ['unknown', True, False]
        ARM_CHECKS = ['arm not in front of torso right', 'arm not in front of torso left', 'arm touches not torso right', 'arm touches not torso left']
        self.header['arm_checks'] = dict(zip(ARM_CHECKS, map(lambda x: ARM_CHECK_VALUES[self.header['arm_checks'] >> (2*x) & 3], [0, 1, 2, 3])))
