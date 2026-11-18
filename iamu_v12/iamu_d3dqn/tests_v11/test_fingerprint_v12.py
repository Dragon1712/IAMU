from __future__ import annotations
import unittest
from v11_protocol.experiment_fingerprint_v12 import semantic_digest,compare_for_resume

class TestFingerprintV12(unittest.TestCase):
    def test_resume_flag_is_runtime_only(self):
        base=dict(schema='v12-scientific-manifest-v1',
                  scientific_config={'margin':0.,'gamma':1.},
                  code_sha256={'trainer':'A'},data_sha256={'case':'B'},
                  task_list_sha256='C',runtime_config={'resume':False,'out':'a'})
        resumed=dict(base,runtime_config={'resume':True,'out':'b'})
        self.assertTrue(compare_for_resume(base,resumed))
        self.assertEqual(semantic_digest(base),semantic_digest(resumed))
    def test_code_and_data_changes_invalidate(self):
        a=dict(schema='v12-scientific-manifest-v1',scientific_config={'gamma':1.},
               code_sha256={'train':'ABC'},data_sha256={'data':'DEF'},
               task_list_sha256='tasks',runtime_config={})
        for field, replacement in [('scientific_config',{'gamma':.95}),
                                    ('code_sha256',{'train':'CHANGED'}),
                                    ('data_sha256',{'data':'CHANGED'}),
                                    ('task_list_sha256','CHANGED')]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                compare_for_resume(a,dict(a,**{field:replacement}))

if __name__=='__main__':unittest.main()
