import unittest

sample_xml_string = """
<!DOCTYPE PickedPoints>
<PickedPoints>
 <DocumentData>
  <DateTime time="16:00:00" date="2014-12-31"/>
  <User name="bodylabs"/>
  <DataFileName name="a_pose.ply"/>
 </DocumentData>
 <point x="0.044259" y="0.467733" z="-0.060032" name="Femoral_epicon_med_lft"/>
<point x="0.017893" y="1.335375" z="0.018390" name="Clavicale_lft"/>
<point x="0.000625" y="1.124424" z="0.080930" name="Substernale"/>
</PickedPoints>
"""

sample_points = {
    'Femoral_epicon_med_lft': [0.044259, 0.467733, -0.060032],
    'Clavicale_lft': [0.017893, 1.335375, 0.018390],
    'Substernale': [0.000625, 1.124424, 0.080930],
}

class TestMeshlabPickedPoints(unittest.TestCase):

    def assertEqualPickedPointXml(self, value, expected):
        # This is fairly naive -- doesn't consider whitespace or that the points
        # could be in different order, and is strict about irrelevant parts like
        #the DateTime. But seems to work okay.
        from lxml import etree, objectify

        value_normalized = etree.tostring(objectify.fromstring(value))
        expected_normalized = etree.tostring(objectify.fromstring(expected))

        self.assertEquals(value_normalized, expected_normalized)

    def test_load(self):
        import StringIO
        from lace.serialization import meshlab_pickedpoints

        sample_f = StringIO.StringIO(sample_xml_string)

        try:
            result = meshlab_pickedpoints.load(sample_f)
        finally:
            sample_f.close()

        self.assertEqual(result, sample_points)

    def test_dump(self):
        import StringIO
        from lace.serialization import meshlab_pickedpoints

        result_f = StringIO.StringIO()

        try:
            meshlab_pickedpoints.dump(sample_points, result_f, mesh_filename='a_pose.ply')
            result_str = result_f.getvalue()
        finally:
            result_f.close()

        self.assertEqualPickedPointXml(result_str, sample_xml_string)


if __name__ == '__main__': # pragma: no cover
    unittest.main()
