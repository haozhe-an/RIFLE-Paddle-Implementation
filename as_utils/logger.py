
import os
import time
import glob
import numpy as np


class LoggerText(object):
    def __init__(self, outdir, splitter=','):
        save_n = 0
        self.file_name = os.path.join(outdir, f'result_{save_n}.out')
        while os.path.exists(self.file_name):
            save_n += 1
            self.file_name = os.path.join(outdir, f'result_{save_n}.out')
            continue

        self.file_write = open(self.file_name, 'w')
        self.file_write.write(str(time.ctime()) + '\n')
        self.splitter = splitter

    def write_once(self, v_stringeable):
        self.file_write.write(str(v_stringeable) + '\n')

    def write_one_step_results(self, step, training_results, test_results=None):
        wrapped_str = f'Training:{step}'
        for r in training_results:
            wrapped_str += f',{r}'
        self.write_once(wrapped_str)

        if test_results is not None:
            wrapped_str = f'TEST:{step}'
            for r in test_results:
                wrapped_str += f',{r}'
            self.write_once(wrapped_str)

    def close(self):
        self.file_write.close()

    def flush(self):
        self.file_write.flush()

    @staticmethod
    def read_all(log_filename, splitter=","):
        f_log = open(log_filename)
        lines = f_log.readlines()
        if len(lines) < 3:  # no hyperp dict
            return
        if 'TEST' not in lines[-1]:  # no TEST data
            return

        args = lines[1].replace(" ", "").replace("Namespace(", "").replace(")", "").replace("\n", "").split(splitter)
        args = {arg.split('=')[0]: arg.split('=')[1] for arg in args}
        # print(args)

        # remove space, and '
        training_names = lines[2][1:-2].replace(" ", "").replace("'", "").split(splitter)
        test_names = lines[3][1:-2].replace(" ", "").replace("'", "").split(splitter)

        training_dict_results = {n: [] for n in ['epoch'] + training_names}
        test_dict_results = {n: [] for n in ['epoch'] + test_names}
        # print(training_dict_results)
        for line in lines:
            if 'Training' in line:
                training_results = line.replace("Training:", "").split(splitter)
                if len(training_names) + 1 != len(training_results):
                    print(f'conflicts found in {log_filename}.')
                    print(len(training_names), len(training_results))

                for i, n in enumerate(training_dict_results.keys()):
                    training_dict_results[n].append(float(training_results[i]))

                continue

            if 'TEST' in line:
                test_results = line.replace("TEST:", "").split(splitter)
                if len(test_names) + 1 != len(test_results):
                    print(f'conflicts found in {log_filename}.')
                    print(len(test_names), len(test_results))

                for i, n in enumerate(test_dict_results.keys()):
                    test_dict_results[n].append(float(test_results[i]))

                continue

        # print(training_dict_results)
        # print(test_dict_results)

        return training_dict_results, test_dict_results, args


def collect_results(outdir, splitter=',', must_have='', cr_threshold=0.0):
    exp_dirs = sorted([d.name for d in os.scandir(outdir) if d.is_dir()])
    # print(exp_dirs)

    filename = f"./{outdir}/{must_have + '_all_results.txt'}"
    collect_results_file = open(filename, 'w')

    for each_exp in exp_dirs:
        each_exp = os.path.join(outdir, each_exp)
        if must_have not in each_exp:
            continue

        each_run_average_performances = []
        each_run_best_performances = []
        all_logfiles = glob.glob(each_exp + '/*.out')
        if len(all_logfiles) == 0:
            continue

        num_nans = 0
        for each_logfile in all_logfiles:
            try:
                training_dict_results, test_dict_results, args = LoggerText.read_all(each_logfile, splitter)
            except EOFError:
                print('EOFError when reading (so skipping)', each_logfile)
                continue
            except ModuleNotFoundError:
                print('ModuleNotFoundError when reading (so skipping)', each_logfile)
                continue
            except ValueError as e:
                print('ValueError:', e)
                if 'nan' in str(e):
                    num_nans += 1
                continue
            except:
                continue

            # full results
            epoch = int(args['epochs'])
            now_epoch = int(test_dict_results['epoch'][-1])

            if now_epoch + 1 != epoch:
                print(f"this one not finished, now {now_epoch}, expected {epoch}: {each_logfile}")
                continue
            if 'precision' not in test_dict_results.keys():
                continue

            # print(each_run, '\t', each_result.test_results['precision'][-1])
            # print(each_run, '\t', max(each_result.test_results['precision']))
            each_run_average_performances.append(test_dict_results['precision'][-1])
            each_run_best_performances.append(max(test_dict_results['precision']))

        if len(each_run_best_performances) == 0 and num_nans == 0:
            continue
        elif len(each_run_best_performances) == 0 and num_nans > 0:
            mean_result = '{4}\t{0:.3f}+-{1:.3f}, {2:.3f}+-{3:.3f}'.format(
                0, 0, 0, 0,
                len(each_run_best_performances)
            )
            print(mean_result + '\t' + each_exp + '\t' + str(num_nans))
            collect_results_file.write(mean_result + '\t' + each_exp + '\t' + str(num_nans) + '\n')
            continue

        if np.mean(each_run_average_performances) < cr_threshold:
            continue

        mean_result = '{4}\t{0:.3f}+-{1:.3f}, {2:.3f}+-{3:.3f}'.format(
            np.mean(each_run_average_performances),
            np.std(each_run_average_performances),
            np.mean(each_run_best_performances),
            np.std(each_run_best_performances),
            len(each_run_best_performances)
        )
        print(mean_result + '\t' + each_exp + '\t' + str(num_nans))
        collect_results_file.write(mean_result + '\t' + each_exp + '\t' + str(num_nans) + '\n')

