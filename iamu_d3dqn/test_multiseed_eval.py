import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPT = pathlib.Path(__file__).with_name("independent_multiseed_eval.py")
spec = importlib.util.spec_from_file_location("iamu_multi_test", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FunctionsTests(unittest.TestCase):
    def test_folder_mapping(self):
        self.assertEqual(m.run_folder("ddqn96", 42), "v082_8workers_sync1_300")
        self.assertEqual(m.run_folder("set_ddqn96", 43), "v09b_set_ddqn_seed43_300")
        self.assertEqual(m.run_folder("d3qn96", 44), "v09a_d3qn_seed44_300")
        self.assertEqual(m.parse_ints("42,43:44"), [42, 43, 44])

    def test_disjoint_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp)
            cases=[]
            for j in range(3):
                p=root / f"{j}.txt"
                p.write_text(f"case-{j}")
                cases.append(str(p))
            out=m.case_hashes(cases[:1],cases[1:2],cases[2:])
            self.assertEqual([len(v) for v in out.values()],[1,1,1])
            with self.assertRaises(ValueError):
                m.case_hashes(cases[:1],cases[1:2],cases[:1])

    def test_cvar_tail_fraction(self):
        self.assertAlmostEqual(m.cvar90_positive_regret([0]*9+[10]), 10)
        self.assertAlmostEqual(m.cvar90_positive_regret([-2,-1,0]), 0)
        self.assertAlmostEqual(m.cvar90_positive_regret([0]*8+[5,10]), 10)

    def test_validate_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/"curve.csv"
            m.save_csv(path,[dict(completed=k,mean_wait=1999-k/100,
                                  evaluated_runs=216) for k in range(25,301,25)])
            self.assertEqual(m.validation_best_score(path), (300,1996.0))


class SyntheticFullRunTests(unittest.TestCase):
    """End-to-end with a stubbed IAMU simulator; not a full real-port validation."""
    def test_end_to_end_single_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            case_dir=root/"cases"; case_dir.mkdir()
            cases=[]
            for j in range(3):
                p=case_dir/f"{j}.txt";p.write_text(f"ports case {j}")
                cases.append(str(p))
            paths=[root/"results"/m.run_folder(label,seed) for label in m.MODEL_SETS["core"] for seed in (42,43)]
            for label in m.MODEL_SETS["core"]:
                for seed in (42,43):
                    arch,width=m.MODELS[label]
                    out=root/"results"/m.run_folder(label,seed)
                    out.mkdir(parents=True)
                    cfg=dict(train_cases=cases[:1],val_cases=cases[1:2],episodes=300,
                             workers=8,validation_workers=8,prefetch_per_worker=1,sync_every=1,
                             scenarios="shock,repeated,small",budgets="2,5,10",train_seed_start=200000,
                             val_seeds="400001:400003",uncertainties="4,8,12",validation_uncertainty=8.,
                             updates=2,clock="union",period=12.,candidate_limit=4,
                             reward_scale=100.,candidate_screen="positive",min_gain=.1,
                             budget_features="absolute",value_features="global",keep_margin=.1,
                             greedy_exploration_prob=.75,lr=.0003,gamma=1.,replay_capacity=50000,
                             batch_size=64,warmup=512,updates_per_step=.25,max_updates_per_episode=64,
                             target_update=250,epsilon_start=1.,epsilon_end=.05,
                             epsilon_decay_episodes=300,eval_every=25,
                             seed=seed,q_network=arch,hidden_dim=width)
                    (out/"manifest.json").write_text(json.dumps(dict(training_args=cfg)))
                    (out/"performance.json").write_text(json.dumps(dict(
                        completed_episodes=300,completed_validations=12,total_validated_runs=2592)))
                    m.save_csv(out/"validation_curve.csv",[dict(
                        completed=k,mean_wait=2000-k/100,
                        evaluated_runs=216) for k in range(25,301,25)])
                    (out/"best.pt").write_text("model weights fixture")

            class FakeAgent:
                def __init__(self, name, seed):
                    self.q_network=m.MODELS[name][0]
                    self.hidden_dim=m.MODELS[name][1]
                    self.keep_margin=.1
                    self.name=name
                    self.seed=seed
                    self.online=types.SimpleNamespace(eval=lambda:None)

                @staticmethod
                def from_checkpoint(p, device):
                    for name in m.MODEL_SETS["core"]:
                        for s in (42,43):
                            if m.run_folder(name,s) in str(p):
                                return FakeAgent(name,s), dict(selected_episode=300,selection_score=1997.)
                    raise AssertionError(p)

            stub=types.ModuleType("iamu_dynamic")
            stub.read_case=lambda path:(path,"port")
            stub.generate_scenario=lambda vessels,port,label,seed,updates_per_vessel,uncertainty:(vessels,label,seed)
            ddqn=types.ModuleType("iamu_dynamic.rl_ddqn")
            ddqn.DoubleDQNAgent=FakeAgent
            exp=types.ModuleType("iamu_dynamic.rl_experiment")
            exp.parse_ints=m.parse_ints
            def fake_episode(instance,budget,policy,agent=None,**kwargs):
                # No edits for keep; model outcomes use architecture and seed.
                cost={"keep":130,"periodic":100,"greedy":105,"ddqn":100}[policy]
                if agent is not None:
                    cost={"ddqn96":100,"ddqn130":97,"set_ddqn96":95}[agent.name]
                    cost += agent.seed-42
                cost+= {"shock":20,"repeated":5,"small":-5}[instance[1]]
                cost+=budget/10 + instance[2]%10
                edits=0 if policy=="keep" else min(2,budget)
                return dict(wait=cost,spent=edits,steps=3,
                            edits_by_eta=edits,edits_by_clock=0)
            exp.run_episode=fake_episode
            with mock.patch.dict(sys.modules, {
                "iamu_dynamic":stub,"iamu_dynamic.rl_ddqn":ddqn,
                "iamu_dynamic.rl_experiment":exp,
            }):
                params=types.SimpleNamespace(mode="core",train_seeds="42,43",eta_seeds="500001:500002",
                    test_cases=cases[2:],results_root=str(root/"results"),workers=1,
                    bootstrap_reps=100,out=str(root/"test_output"),dry_run=False)
                m.test_main(params)
            out=root/"test_output"
            runs=m.load_csv(out/"independent_test_runs.csv")
            self.assertEqual(len(runs), 1*3*2*3*(3+3*2))  # 18 paired groups, 9 policies per group
            pairs=m.load_csv(out/"paired_differences.csv")
            self.assertEqual(len(pairs), 18*2*2) # 2 model comparisons, 2 training seeds
            contrasts=m.load_csv(out/"architecture_bootstrap.csv")
            self.assertEqual(len(contrasts),2)
            diff={r["contrast"]:float(r["mean_wait_diff"]) for r in contrasts}
            self.assertAlmostEqual(diff["set_ddqn96 - ddqn96"],-5.)
            self.assertAlmostEqual(diff["set_ddqn96 - ddqn130"],-2.)
            man=json.loads((out/"audit_manifest.json").read_text())
            self.assertEqual(man["total_test_runs"], 162)
            # Never overwrite test outputs accidentally.
            with mock.patch.dict(sys.modules, {
                "iamu_dynamic":stub,"iamu_dynamic.rl_ddqn":ddqn,
                "iamu_dynamic.rl_experiment":exp,
            }),self.assertRaises(FileExistsError):
                m.test_main(params)


if __name__=="__main__":
    unittest.main()
