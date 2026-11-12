import tempfile
import unittest
from pathlib import Path

class LegacyAdapterContractTests(unittest.TestCase):
    def test_probe_compilation_and_frozen_checkpoint(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch not installed')
        from iamu_dynamic.legacy_erldde import build_probe,FrozenERLDDEHints
        from iamu_dynamic import Vessel,Port
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);root=d/'upstream';root.mkdir()
            # Stub compiler fixture ONLY. Does not claim to test author's simulator.
            (root/'BACAP_env.cpp').write_text(r'''
#include <vector>
#include <tuple>
#include <cstring>
class BerthOptimizationEnvironment {
 public:
 std::vector<std::tuple<int,double>> arrival_vessel;
 BerthOptimizationEnvironment(const char*) { arrival_vessel.push_back({0,0.5}); }
};
struct Array2D {double data[30][8];};
struct StepResult {Array2D state;double reward;bool done;};
extern "C" {
 void* create_env(const char* f) {return new BerthOptimizationEnvironment(f);}
 void delete_env(BerthOptimizationEnvironment* p) {delete p;}
 void reset(BerthOptimizationEnvironment*, Array2D* a) {std::memset(a,0,sizeof(*a));}
 void step(BerthOptimizationEnvironment*,const double*,StepResult* r) {
   std::memset(r,0,sizeof(*r));r->done=true;
 }
}
''',encoding='utf-8')
            (root/'model.py').write_text('''
import numpy as np
import torch
class ES(torch.nn.Module):
    def __init__(self):
        super().__init__();self.w=torch.nn.Parameter(torch.ones(1))
    def forward(self,state):
        return np.array([0,0.1,0.0,0.1,0.5])
''',encoding='utf-8')
            ck=d/'test.pth'
            torch.save({'w':torch.ones(1)},ck)
            lib=build_probe(root,d/'out')
            hint=FrozenERLDDEHints(root,ck,lib)([Vessel(4,2,30,300,3)],Port(cranes=4),{4:2})
            self.assertEqual(hint.rank,{4:0})
            self.assertAlmostEqual(hint.position[4],300)
            self.assertEqual(hint.backend,'frozen_erldde')

if __name__=='__main__':unittest.main()
