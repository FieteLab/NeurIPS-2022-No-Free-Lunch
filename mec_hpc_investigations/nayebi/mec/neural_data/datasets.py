import os
import copy
import pickle, h5py
import numpy as np
from mec_hpc_investigations.core.default_dirs import CAITLIN2D_WITH_INERTIAL, CAITLIN2D_WITHOUT_INERTIAL, CAITLIN1D_VR, REWARD_BASE_DIR, CAITLIN2D_PLACE_CELL, CAITLIN2D_HPC
from mec_hpc_investigations.neural_data.utils import loadmat, populate_session_tetrode_id

class DatasetBase(object):
    def __init__(self, data_path,
                 protocol=2):
        self.data_path = data_path
        self.protocol = protocol
        self.packaged_data = None

    def fetch(self):
        #TODO: put data on s3 once internet connection stable
        #add s3url argument
        #add s3 stuff to utils to fetch data if not in data_path
        pass

    def save_data(self, save_path):
        assert self.packaged_data is not None
        pickle.dump(self.packaged_data,
                    open(save_path, "wb"),
                    protocol=self.protocol)

class CaitlinDatasetWithInertial(DatasetBase):
    def __init__(self, data_path=None, **kwargs):
        if data_path is None:
            data_path = CAITLIN2D_WITH_INERTIAL
        super(CaitlinDatasetWithInertial, self).__init__(data_path=data_path, **kwargs)

    def load_data(self):
        self.fetch()
        self.ld_data = loadmat(self.data_path)
        self.num_cells = len(self.ld_data["cell_info"])

    def package_data(self):
        '''
        Returns the packaged Python version of this dataset
        (np.recarray for now since dataset is non-rectangular across cells)
        '''
        self.load_data()
        packaged_data = {}
        for c_idx in range(self.num_cells):
            curr_d = populate_session_tetrode_id(self.ld_data["cell_info"][c_idx].__dict__)
            curr_d.pop('_fieldnames')
            for k, v in curr_d.items():
                if c_idx == 0:
                    packaged_data[k] = [v]
                else:
                    packaged_data[k].append(v)

        self.packaged_data = np.core.records.fromarrays(packaged_data.values(),
                                                        names=list(packaged_data.keys()))

class CaitlinDatasetWithoutInertial(DatasetBase):
    def __init__(self, data_path=None, **kwargs):
        if data_path is None:
            data_path = CAITLIN2D_WITHOUT_INERTIAL
        super(CaitlinDatasetWithoutInertial, self).__init__(data_path=data_path, **kwargs)

    def load_data(self):
        self.fetch()
        self.ld_data = h5py.File(self.data_path, 'r')
        self.num_cells = len(self.ld_data["cell_info"]["cell_id"])

    def package_data(self):
        '''
        Returns the packaged Python version of this dataset
        (np.recarray for now since dataset is non-rectangular across cells)
        '''
        self.load_data()
        packaged_data = {}
        for c_idx in range(self.num_cells):
            curr_d = {}
            for k in list(self.ld_data["cell_info"].keys()):
                curr_entry = self.ld_data["cell_info"][k][c_idx]
                assert(len(curr_entry) == 1)
                curr_obj = self.ld_data[curr_entry[0]][:]
                if k in ["animal_id", "cell_id"]: # strings treated differenly
                    curr_obj = ''.join(chr(j) for j in curr_obj)
                    curr_d[k] = curr_obj
                else:
                    curr_d[k] = np.squeeze(curr_obj)
            curr_d = populate_session_tetrode_id(curr_d)
            for k, v in curr_d.items():
                if c_idx == 0:
                    packaged_data[k] = [v]
                else:
                    packaged_data[k].append(v)

        self.packaged_data = np.core.records.fromarrays(packaged_data.values(),
                                                        names=list(packaged_data.keys()))

class CaitlinDataset1DVR(DatasetBase):
    def __init__(self, data_path=None, sess_idxs=range(1, 12), **kwargs):
        if data_path is None:
            data_path = CAITLIN1D_VR
        if isinstance(sess_idxs, int):
            sess_idxs = [sess_idxs]
        super(CaitlinDataset1DVR, self).__init__(data_path=data_path, **kwargs)
        self.session_filenames = [f"cell_info_session{s}.mat" for s in sess_idxs]

    def load_data(self):
        self.fetch()
        self.ld_data = {}
        for s_f in self.session_filenames:
            self.ld_data[s_f] = h5py.File(os.path.join(self.data_path, s_f), 'r')
        self.num_cells = {}
        for s_f in self.session_filenames:
            self.num_cells[s_f] = len(self.ld_data[s_f]["cell_info"]["cell_id"])

    def package_data(self):
        '''
        Returns the packaged Python version of this dataset
        (np.recarray for now since dataset is non-rectangular across cells)
        '''
        self.load_data()
        packaged_data = {}
        num_total_cells = 0
        # loop through each .mat file
        for s_f_idx, s_f in enumerate(self.session_filenames):
            num_total_cells += self.num_cells[s_f]
            for c_idx in range(self.num_cells[s_f]):
                # construct dictionary for each cell
                curr_d = {"session_filename": s_f}
                for k in list(self.ld_data[s_f]["cell_info"].keys()):
                    curr_entry = self.ld_data[s_f]["cell_info"][k][c_idx]
                    assert(len(curr_entry) == 1)
                    curr_obj = self.ld_data[s_f][curr_entry[0]][:]
                    if k in ["animal_id", "cell_id", "session_id"]: # strings processed differenly
                        curr_obj = ''.join(chr(j) for j in curr_obj)
                        curr_d[k] = curr_obj
                    else:
                        curr_d[k] = np.squeeze(curr_obj)
                # append each cell's value to the corresponding attribute to master dictionary
                for k, v in curr_d.items():
                    if (s_f_idx == 0) and (c_idx == 0):
                        packaged_data[k] = [v]
                    else:
                        packaged_data[k].append(v)

            # sanity check that we got all cells so far
            assert(len(np.unique(packaged_data["cell_id"])) == num_total_cells)
            for k in packaged_data.keys():
                assert(len(packaged_data[k]) == num_total_cells)

        self.master_dict = packaged_data
        self.packaged_data = np.core.records.fromarrays(packaged_data.values(),
                                                        names=list(packaged_data.keys()))


class RewardDataset(DatasetBase):
    def __init__(self,
                 dataset='task',
                 base_path=None,
                 nbins=30,
                 smooth_std=1,
                 dt=0.02,
                 start_frac=None,
                 end_frac=None,
                 verbose=False,
                 **kwargs):

        import pandas as pd

        self.dataset = dataset
        self.nbins = nbins
        self.smooth_std = smooth_std
        self.dt = dt
        self.start_frac = start_frac if start_frac is not None else 0
        self.end_frac = end_frac if end_frac is not None else 1 
        self.verbose = verbose

        if (dataset != 'task') and (dataset != 'free_foraging'):
            raise ValueError

        if base_path is None:
            base_path = REWARD_BASE_DIR

        self.metadata = pd.read_csv(os.path.join(base_path, f"{dataset}_cell_info_table.csv"))
        data_path = os.path.join(base_path, f"{dataset}_cell_data.h5")
        super(RewardDataset, self).__init__(data_path=data_path, **kwargs)

    def load_data(self):
        self.fetch()
        f = h5py.File(self.data_path, "r")
        self.n_sessions = len(f)
        if self.verbose:
            print(f'Number of sessions in data = {self.n_sessions}')

        self.n_neurons = 0
        self.spikes = {}
        self.behav_data = {}
        for session_id in range(self.n_sessions):
            self.behav_data[session_id] = f[f'session_{session_id+1}/behavioral_data'][:]
            # it is in ms originally, so we convert to seconds
            diff_t = np.diff(self.behav_data[session_id][:, 0]/1000.0)
            # sometimes this can be 0.019999...
            assert(np.isclose(diff_t, self.dt*np.ones_like(diff_t), rtol=1e-1).all())
            # cells by time
            self.spikes[session_id] = f[f'session_{session_id+1}/neural_data'][:].T
            self.n_neurons += self.spikes[session_id].shape[0]

        # close h5 file
        f.close()
        if self.verbose:
            print(f'Number of neurons = {self.n_neurons}')

    def get_arena_bins(self):
        """Get x,y bins for the reward dataset.
        Mininum and maximum across animals per session"""
        body_x_min = np.amin([np.nanmin(self.behav_data[session_id][:, 1]) for session_id in self.behav_data.keys()])
        body_x_max = np.amax([np.nanmax(self.behav_data[session_id][:, 1]) for session_id in self.behav_data.keys()])

        body_y_min = np.amin([np.nanmin(self.behav_data[session_id][:, 2]) for session_id in self.behav_data.keys()])
        body_y_max = np.amax([np.nanmax(self.behav_data[session_id][:, 2]) for session_id in self.behav_data.keys()])

        arena_x_bins = np.linspace(body_x_min, body_x_max, self.nbins+1, endpoint=True)
        arena_y_bins = np.linspace(body_y_min, body_y_max, self.nbins+1, endpoint=True)
        return arena_x_bins, arena_y_bins

    def compute_rate_maps(self):

        import scipy
        from scipy.stats import binned_statistic_2d

        rate_maps = []
        for session_id in range(self.n_sessions):
            curr_body_x = self.behav_data[session_id][:, 1]
            curr_body_y = self.behav_data[session_id][:, 2]
            curr_spikes = self.spikes[session_id]
            assert(len(curr_body_x) == len(curr_body_y))
            assert(len(curr_body_x) == curr_spikes.shape[1])
            num_time = len(curr_body_x)
            start_idx = (int)(np.ceil(self.start_frac*num_time))
            end_idx = (int)(np.ceil(self.end_frac*num_time))
            if self.verbose:
                print(f'Start idx: {start_idx} End idx: {end_idx}')
            curr_body_x = curr_body_x[start_idx:end_idx]
            curr_body_y = curr_body_y[start_idx:end_idx]
            curr_spikes = curr_spikes[:, start_idx:end_idx]
            curr_binned_frs = binned_statistic_2d(x=curr_body_x,
                                                  y=curr_body_y,
                                                  values=curr_spikes,
                                                  statistic="mean",
                                                  bins=[self.arena_x_bins, self.arena_y_bins],
                                                  expand_binnumbers=True)
            curr_binned_avg_frs = curr_binned_frs.statistic*(1.0/self.dt)
            # x by y by cells
            curr_binned_avg_frs = np.transpose(curr_binned_avg_frs, axes=(1, 2, 0))
            rate_maps.append(curr_binned_avg_frs)
        rate_maps = np.concatenate(rate_maps, axis=-1)
        assert(rate_maps.shape[-1] == self.n_neurons)

        if self.smooth_std is not None:
            smooth_arr = []
            for c_idx in range(self.n_neurons):
                curr_binned_avg_frs = rate_maps[:, :, c_idx]
                curr_binned_avg_frs = scipy.ndimage.gaussian_filter(input=np.nan_to_num(curr_binned_avg_frs),
                                                                     sigma=self.smooth_std, mode="constant")
                smooth_arr.append(curr_binned_avg_frs)
            rate_maps = np.stack(smooth_arr, axis=-1)

        return rate_maps

    def aggregate_responses(self):
        assert self.packaged_data is not None
        spec_resp_agg = {150: {}}
        neuron_counter = 0
        for animal in np.unique(self.metadata['subjects']):
            spec_resp_agg[150][animal] = {}
            animal_cell_idx = (self.metadata['subjects'] == animal)
            spec_resp_agg[150][animal]['resp'] = self.packaged_data[:, :, animal_cell_idx]
            spec_resp_agg[150][animal]['cell_ids'] = np.arange(self.n_neurons)[animal_cell_idx]
            neuron_counter += len(spec_resp_agg[150][animal]['cell_ids'])
        # ensures we went through all neurons
        assert(neuron_counter == self.n_neurons)
        return spec_resp_agg

    def package_data(self):
        '''
        Returns the packaged Python version of this dataset
        '''
        self.load_data()
        self.arena_x_bins, self.arena_y_bins = self.get_arena_bins()
        self.packaged_data = self.compute_rate_maps()
        self.spec_resp_agg = self.aggregate_responses()


class PlaceCellDataset(DatasetBase):
    def __init__(self,
                 data_path=None,
                 bin_cm=5.0,
                 smooth_std=1,
                 dt=0.02,
                 metadata=None,
                 verbose=False,
                 **kwargs):

        import pandas as pd

        self.bin_cm = bin_cm
        self.smooth_std = smooth_std
        self.dt = dt
        self.verbose = verbose

        if data_path is None:
            data_path = CAITLIN2D_PLACE_CELL
        self.metadata = metadata
        if self.metadata is None:
            # same as "Place Cells.csv" but with two animal names corrected in the filepath"
            self.metadata = pd.read_csv(os.path.join(data_path, "PlaceCells_Aran.csv"), encoding="Windows-1252")
        super(PlaceCellDataset, self).__init__(data_path=data_path, **kwargs)

    def parse_session(self, s):
        l = s.split("\\")
        assert(len(l) == 3)
        curr_sess = l[2]
        curr_animal = l[1]
        return curr_sess, curr_animal

    def load_data(self, animal_delim="/"):
        self.fetch()
        self.unparsed_sessions = list(self.metadata["Session(s)"])
        self.n_neurons = len(self.unparsed_sessions)
        self.animals = list(self.metadata["Mouse"])
        self.tetrodes = list(self.metadata["Tetrode"])
        self.units = list(self.metadata["Unit"])
        self.boxes = list(self.metadata["Box"])
        # we grab all the cells relevant to each session so we can compute min and max position bounds across them to rescale by the box size
        sess_to_bounds = {}
        sess_to_boxes = {}
        cell_resp = {}
        visited_cells = []
        for s_idx, s in enumerate(self.unparsed_sessions):
            curr_sess, curr_animal = self.parse_session(s)
            curr_box = self.boxes[s_idx]
            if curr_sess not in sess_to_bounds.keys():
                sess_to_bounds[curr_sess] = {"x_pos_min": np.inf, "x_pos_max": -np.inf, "y_pos_min": np.inf, "y_pos_max": -np.inf}
                sess_to_boxes[curr_sess] = self.boxes[s_idx]
            curr_pos = loadmat(os.path.join(self.data_path, f"{curr_animal}{animal_delim}{curr_sess}_pos.mat"))
            x_pos = (curr_pos["posx"] + curr_pos["posx2"])/2.0
            y_pos = (curr_pos["posy"] + curr_pos["posy2"])/2.0
            # get min and max for each animal in a given session
            x_pos_currmin = np.nanmin(x_pos)
            x_pos_currmax = np.nanmax(x_pos)
            y_pos_currmin = np.nanmin(y_pos)
            y_pos_currmax = np.nanmax(y_pos)
            # then min and max across animals for that session to get session environment absolute bounds (as camera position might vary per session)
            if x_pos_currmin < sess_to_bounds[curr_sess]["x_pos_min"]:
                sess_to_bounds[curr_sess]["x_pos_min"] = x_pos_currmin
            if x_pos_currmax > sess_to_bounds[curr_sess]["x_pos_max"]:
                sess_to_bounds[curr_sess]["x_pos_max"] = x_pos_currmax
            if y_pos_currmin < sess_to_bounds[curr_sess]["y_pos_min"]:
                sess_to_bounds[curr_sess]["y_pos_min"] = y_pos_currmin
            if y_pos_currmax > sess_to_bounds[curr_sess]["y_pos_max"]:
                sess_to_bounds[curr_sess]["y_pos_max"] = y_pos_currmax
            # sanity check that all sessions have one unique box size across animals
            assert(sess_to_boxes[curr_sess] == self.boxes[s_idx])
            if curr_box not in cell_resp.keys():
                cell_resp[curr_box] = {}
            if curr_animal not in cell_resp[curr_box].keys():
                cell_resp[curr_box][curr_animal] = {}
            curr_cell = f"{curr_sess}_T{self.tetrodes[s_idx]}C{self.units[s_idx]}"
            # making sure we don't come across the same cell and rewrite things
            curr_cell_nm = f"{curr_animal}_{curr_cell}"
            if curr_cell_nm in visited_cells:
                raise ValueError
            visited_cells.append(curr_cell_nm)
            if curr_sess not in cell_resp[curr_box][curr_animal].keys():
                cell_resp[curr_box][curr_animal][curr_sess] = {}
            cell_resp[curr_box][curr_animal][curr_sess][curr_cell] = {}
            cell_resp[curr_box][curr_animal][curr_sess][curr_cell]["t_pos"] = curr_pos["post"]
            cell_resp[curr_box][curr_animal][curr_sess][curr_cell]["x_pos"] = x_pos
            cell_resp[curr_box][curr_animal][curr_sess][curr_cell]["y_pos"] = y_pos
            cell_resp[curr_box][curr_animal][curr_sess][curr_cell]["spt"] = loadmat(os.path.join(self.data_path, f"{curr_animal}{animal_delim}{curr_cell}.mat"))["cellTS"]
        # sanity check that we have accounted for all cells
        assert(len(visited_cells) == self.n_neurons)
        return cell_resp, sess_to_bounds

    def get_arena_bins(self, curr_box):
        # ensures <= self.bin_cm, never greater
        nbins = (int)(np.ceil(((float)(curr_box))/self.bin_cm))
        if self.verbose:
            print(f"Using {nbins} bins for arena of size {curr_box} cm on each side")
        arena_x_bins = np.linspace(0, curr_box, nbins+1, endpoint=True)
        arena_y_bins = copy.deepcopy(arena_x_bins)
        return arena_x_bins, arena_y_bins

    def compute_rate_maps(self, spt, x_pos, y_pos,
                          x_pos_min, x_pos_max,
                          y_pos_min, y_pos_max,
                          t_pos, curr_box,
                          arena_x_bins, arena_y_bins):
        """Compute rate maps for a single cell"""
        import scipy
        from scipy.stats import binned_statistic_2d

        # we rescale between 0 and 1, then multiply by box size
        x_cm = (x_pos - x_pos_min)
        x_cm *= (((float)(curr_box))/(x_pos_max - x_pos_min))
        y_cm = (y_pos - y_pos_min)
        y_cm *= (((float)(curr_box))/(y_pos_max - y_pos_min))
        curr_spt = np.round(spt/self.dt)*self.dt
        # temporal sanity checks that everything is at the temporal resolution described and in sorted order
        assert((~np.isnan(t_pos)).all())
        assert((~np.isnan(curr_spt)).all())
        assert(np.array_equal(t_pos, np.sort(t_pos)))
        assert(np.array_equal(curr_spt, np.sort(curr_spt)))
        diff_t = np.diff(t_pos)
        assert(np.isclose(diff_t, self.dt*np.ones_like(diff_t)).all())
        curr_t_indices = []
        for spt_item in curr_spt:
            # for each spike time, find the closest time (faster than two for loops before for unrounded MEC 1d data)
            time_item_idx = np.argmin(np.abs(t_pos - spt_item))
            curr_t_indices.append(time_item_idx)
        # generally find the counts of each spike time
        _, c = np.unique(curr_spt, return_counts=True)
        sp_counts = np.zeros(t_pos.shape)
        # associate the times with the counts
        sp_counts[np.unique(curr_t_indices)] = c
        assert(np.sum(sp_counts) == curr_spt.shape[0])

        # this is equivalent to summing the sp_counts above for values in each bin
        # then dividing by the counts in each position bin * (dt, which is the time it takes in each position)
        curr_binned_frs = binned_statistic_2d(x=x_cm,
                                              y=y_cm,
                                              values=sp_counts,
                                              statistic="mean",
                                              bins=[arena_x_bins, arena_y_bins],
                                              expand_binnumbers=True)
        curr_binned_avg_frs = curr_binned_frs.statistic*(1.0/self.dt)

        if self.smooth_std is not None:
            curr_binned_avg_frs = scipy.ndimage.gaussian_filter(input=np.nan_to_num(curr_binned_avg_frs),
                                                                 sigma=self.smooth_std, mode="constant")

        return curr_binned_avg_frs

    def aggregate_responses(self, cell_resp, sess_to_bounds):
        """We purposefully make cell_resp and the bounds arguments to allow for aggregating the permutations of spike times for spatial information"""
        num_total_cells = 0
        num_cells_arr = 0
        spec_resp_agg = {}
        for curr_box in cell_resp.keys():
            spec_resp_agg[curr_box] = {}
            arena_x_bins, arena_y_bins = self.get_arena_bins(curr_box)
            if self.verbose:
                print(f"Arena x bins: {arena_x_bins}")
                print(f"Arena y bins: {arena_y_bins}")
            for curr_animal in cell_resp[curr_box].keys():
                spec_resp_agg[curr_box][curr_animal] = {}
                curr_animal_resp = []
                cell_ids = []
                for curr_sess in cell_resp[curr_box][curr_animal].keys():
                    for curr_cell in cell_resp[curr_box][curr_animal][curr_sess].keys():
                        # the cell id already includes the session, so no need to include it
                        assert(curr_sess in curr_cell)
                        cell_ids.append(curr_cell)
                        rm_kwargs = copy.deepcopy(cell_resp[curr_box][curr_animal][curr_sess][curr_cell])
                        rm_kwargs.update(sess_to_bounds[curr_sess])
                        rm_kwargs["curr_box"] = curr_box
                        rm_kwargs["arena_x_bins"] = arena_x_bins
                        rm_kwargs["arena_y_bins"] = arena_y_bins
                        ret_val = self.compute_rate_maps(**rm_kwargs)
                        curr_animal_resp.append(ret_val)
                        num_total_cells += 1

                spec_resp_agg[curr_box][curr_animal] = {"resp": np.stack(curr_animal_resp, axis=-1),
                                                        "cell_ids": np.array(cell_ids)}
                assert(len(spec_resp_agg[curr_box][curr_animal]["cell_ids"].shape) == 1)
                assert(spec_resp_agg[curr_box][curr_animal]["resp"].shape[-1] == spec_resp_agg[curr_box][curr_animal]["cell_ids"].shape[0])
                num_cells_arr += spec_resp_agg[curr_box][curr_animal]["resp"].shape[-1]

        # sanity check we have accounted for all cells
        assert(num_total_cells == self.n_neurons)
        assert(num_cells_arr == self.n_neurons)
        return spec_resp_agg

    def package_data(self):
        '''
        Returns the packaged Python version of this dataset
        '''
        self.cell_resp, self.sess_to_bounds = self.load_data()
        self.spec_resp_agg = self.aggregate_responses(cell_resp=self.cell_resp,
                                                      sess_to_bounds=self.sess_to_bounds)

class CaitlinHPCDataset(PlaceCellDataset):
    def __init__(self,
                 data_path=None,
                 bin_cm=5.0,
                 smooth_std=1,
                 dt=0.02,
                 verbose=False,
                 **kwargs):

        import pandas as pd

        self.bin_cm = bin_cm
        self.smooth_std = smooth_std
        self.dt = dt
        self.verbose = verbose

        if data_path is None:
            data_path = CAITLIN2D_HPC
        metadata = pd.read_excel(os.path.join(CAITLIN2D_HPC, "Final place cell dataset.xlsx"), engine='openpyxl')
        # select only wildtype cells
        metadata = metadata[metadata["Genotype"] == 1]
        super(CaitlinHPCDataset, self).__init__(data_path=data_path, 
                                                metadata=metadata,
                                                **kwargs)

    def parse_session(self, s):
        l = s.split("\\")
        curr_sess = l[-1]
        curr_animal = l[-2]
        return curr_sess, curr_animal

    def package_data(self):
        '''
        Returns the packaged Python version of this dataset
        '''
        self.cell_resp, self.sess_to_bounds = self.load_data(animal_delim="_")
        self.spec_resp_agg = self.aggregate_responses(cell_resp=self.cell_resp,
                                                      sess_to_bounds=self.sess_to_bounds)
