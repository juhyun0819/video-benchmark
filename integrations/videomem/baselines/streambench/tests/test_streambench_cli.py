import json
import subprocess
import sys
import unittest

class CLITests(unittest.TestCase):
    def test_method_registry_is_honest_about_missing_baselines(self):
        result=subprocess.run([sys.executable,'-m','baselines.streambench','list-methods'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        methods=json.loads(result.stdout)
        self.assertEqual(methods['rekv']['status'],'adapter_available')
        self.assertEqual(methods['aks']['status'],'adapter_available')
        self.assertEqual(methods['streamforest']['status'],'adapter_available')
        self.assertEqual(methods['hermes']['status'],'not_in_checkout')

    def test_sweep_rejects_nonpositive_budget_before_loading_models(self):
        result=subprocess.run([sys.executable,'-m','baselines.streambench','sweep','--config','missing.json','--manifest','missing.jsonl','--output','unused','--budgets-mib','0'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('positive',result.stderr)

if __name__=='__main__':unittest.main()
