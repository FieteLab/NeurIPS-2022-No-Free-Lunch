% to run this file:
% 1. login to sherlock, and cd into the directory containing this file. 
% (Make sure save_fn is set to whichever dataset you want.)
% 'task' refers to the dataset with the reward location
% 'free_foraging' is the free foraging control with no rewards.
% Next, type the following:
% 2. sdev
% 3. ml load matlab
% 4. matlab -nodisplay < load_wb_kh_all_cells.m

% save_fn = 'free_foraging';
save_fn = 'task';
data_info_dir = '/scratch/groups/yamins/mec_data/reward_data/';
file_name = 'wb_kh_data.mat';

load(strcat(data_info_dir,file_name));

data_path = '/oak/stanford/groups/giocomo/export/data/Users/WButler/D254 Neuralynx';
all_file_ids = {};
for ii = 1:length(cell_info_trained)
    if strcmp(save_fn, 'free_foraging')
        all_file_ids{ii} = cell_info_trained(ii).best_free;
    elseif strcmp(save_fn, 'task')
        all_file_ids{ii} = cell_info_trained(ii).best_task;
    else
        error('wrong filename!');
    end
end
% loads only MEC cells
file_ids = all_file_ids([cell_info_trained.in_mec]==1);
file_ids = file_ids(:);

n_units = length(file_ids);

% pre-allocate file/unit information
subjects = cell(n_units,1);
dates = cell(n_units,1);
times = cell(n_units,1);
channels = cell(n_units,1);
sessions = cell(n_units,1);
sessions_id = zeros(n_units,1);

session_counter = 0;
for ii = 1: n_units
    % deconstruct file id into components
    temp = strsplit(file_ids{ii},'_');
    subjects(ii) = temp(1);
    dates(ii) = temp(2);
    times(ii) = temp(3);
    channels(ii) = temp(4);
    sessions{ii} =  strcat(dates{ii},'_',times{ii});

    % assign a different id to each unique session
    if ii == 1
        session_counter = session_counter + 1;
        sessions_id(ii) = session_counter;
    else
        session_id_match = find(ismember(sessions(1:(ii-1)),sessions(ii)),1);
        if isempty(session_id_match)
            session_counter = session_counter + 1;
            sessions_id(ii) = session_counter;
        else
            sessions_id(ii) = sessions_id(session_id_match);
        end
    end
end
n_sessions = max(sessions_id);
% count number of units by session
sessions_n_units = histcounts(sessions_id,1:(n_sessions+1));

session_units_id = zeros(n_units,1);
for session_id = 1:n_sessions
    session_units_id(sessions_id==session_id) = (1:sessions_n_units(session_id))-1;
end

% session info table
units = (1:n_units)';
unit_info_table = table(units, sessions_id, session_units_id, subjects, ...
    dates, times, channels, sessions, file_ids);
writetable(unit_info_table,fullfile(data_info_dir, strcat(save_fn, '_cell_info_table.csv')))

% constants
samp_rate = 32000; % data sampling rate
frame_rate = 30; % frames per sec / 33ms
resamp_rate = 50; % samps per sec / 20ms
sec_ms = 1000;

save_path = data_info_dir;
save_file_name = fullfile(data_info_dir, strcat(save_fn, '_cell_data.h5'));

%%
% h5 won't allow creating of same unit names.
if isfile(save_file_name)
    delete(save_file_name)
end
wb = waitbar(0, 'Start of unit conversion to h5.');

for session_id = 1:n_sessions
    waitbar(session_id/n_sessions, wb, sprintf('Processing session # %d',session_id))

    unit_ids = find(sessions_id==session_id)';
    n_session_units = length(unit_ids);

    unit_counter = 1;
    for unit = unit_ids
        if unit_counter==1
            subject = subjects{unit};
            session = sessions{unit};
            date = sessions{unit};
            time = sessions{unit};

            file_path = fullfile(data_path, subject, session);

            % behavioral data
            xy_fn = strcat(subject,'_',session, '.mat');
            xy =load(fullfile(file_path,xy_fn));

            % resample the data and create arrays
            t0 = xy.Timestamps(1);
            tE = xy.Timestamps(end);

            n_samps = length(xy.Timestamps);
            total_time_secs = n_samps/frame_rate;

            time_orig_ms = linspace(0,total_time_secs*sec_ms,n_samps);

            n_resamp = int32(total_time_secs*resamp_rate);
            time_resamp_ms = linspace(0,total_time_secs*sec_ms, n_resamp);

            % finally resample x,y and angle
            x_resamp = interp1(time_orig_ms,xy.x,time_resamp_ms, 'nearest');
            y_resamp = interp1(time_orig_ms,xy.y,time_resamp_ms, 'nearest');
            ang_resamp = interp1(time_orig_ms,xy.y,time_resamp_ms, 'nearest');

            % behavior_data
            behavior_data = single([time_resamp_ms; x_resamp; y_resamp; ang_resamp]);

            % find conversion between sample values and seconds
            samp_vals_secs = (tE-t0)/total_time_secs;

            % pre-allocate spike array
            binned_spikes = zeros(n_session_units, n_resamp);

            % save behavioral data and attributes to h5
            h5create(save_file_name, strcat('/session_',string(session_id),'/behavioral_data'), size(behavior_data), 'Datatype','single');
            h5write(save_file_name, strcat('/session_',string(session_id),'/behavioral_data'), behavior_data);
            h5writeatt(save_file_name, strcat('/session_',string(session_id),'/behavioral_data'), 'subject', subject);
            h5writeatt(save_file_name, strcat('/session_',string(session_id),'/behavioral_data'), 'date', date);
            h5writeatt(save_file_name, strcat('/session_',string(session_id),'/behavioral_data'), 'time', time);
            h5writeatt(save_file_name, strcat('/session_',string(session_id),'/behavioral_data'), 'session', session);
            h5writeatt(save_file_name, strcat('/session_',string(session_id),'/behavioral_data'), 'orig_file', xy_fn);
            h5writeatt(save_file_name, strcat('/session_',string(session_id),'/behavioral_data'), 'samp_rate', resamp_rate);
            h5writeatt(save_file_name, strcat('/session_',string(session_id),'/behavioral_data'), 'vars', 'time-x-y-ang');

        end

        % load spike file and process
        spk_fn = strcat(file_ids{unit},'.mat');
        spk_times = load(fullfile(file_path,spk_fn));

        % convert samples to ms and bin
        spk_times_ms = (spk_times.cellTS-t0)/samp_vals_secs*sec_ms;
        [bin_spk,~] = histcounts(spk_times_ms, ...
            [time_resamp_ms,time_resamp_ms(end)+sec_ms/resamp_rate]);
        binned_spikes(unit_counter,:) = bin_spk;

        unit_counter = unit_counter + 1;
    end

    % convert to single
    binned_spikes = single(binned_spikes);

    % save data and attributes to h5
    h5create(save_file_name, strcat('/session_',string(session_id),'/neural_data'), size(binned_spikes), 'Datatype','single');
    h5write(save_file_name, strcat('/session_',string(session_id),'/neural_data'), binned_spikes);

    h5writeatt(save_file_name, strcat('/session_',string(session_id),'/neural_data'), 'samp_rate', resamp_rate);
    h5writeatt(save_file_name, strcat('/session_',string(session_id),'/neural_data'), 'n_units', n_session_units);

    unit_counter = 1;
    for unit = unit_ids
        h5writeatt(save_file_name, strcat('/session_',string(session_id),'/neural_data'),...
            strcat('orig_file_unit_',string(unit_counter)), file_ids{unit});
        unit_counter = unit_counter + 1;
    end
end

close(wb)
