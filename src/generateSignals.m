function [] = generateSignals(frames_per_mod_type, MAX_PPM, Path, FolderName, varargin) 
    %Parameters (* = Required): *frames_per_mod_type, *MAX_PPM, *Path, hard_set_offset, freq_scale_factor, SPF, SPS ) 
    % Creating Default values for parameters: spf, sps, hardsetppm, str_sq
        % Default values
    defaultSetPPMOffset = 0;
    defaultFreqScaleFactor = 1;
    defaultSPF = 1024;
    defaultSPS = 8;
    defaultChannel = 0; %0 = Rician Fading, 1 = Rayleigh Fading
    
    % Input parser setup
    p = inputParser;
    addRequired(p, 'frames_per_mod_type');
    addRequired(p, 'MAX_PPM');
    addRequired(p, 'Path');
    addRequired(p, 'FolderName');
    addParameter(p, 'hard_set_offset', defaultSetPPMOffset, @isnumeric);
    addParameter(p, 'freq_scale_factor', defaultFreqScaleFactor, @isnumeric);
    addParameter(p, 'spf', defaultSPF, @isnumeric);
    addParameter(p, 'sps', defaultSPS, @isnumeric);
    addParameter(p, 'channelType', defaultChannel, @isnumeric); 
    % Parse inputs
    parse(p, frames_per_mod_type, MAX_PPM, Path, FolderName, varargin{:});

    % Assign parsed variables
    hard_set_offset = p.Results.hard_set_offset;
    freq_scale_factor = p.Results.freq_scale_factor;
    spf = p.Results.spf;
    sps = p.Results.sps;
    channelType = p.Results.channelType;


    fs = 200e3;             % Sample rate
    fc = [902e6 100e6];     % Center frequencies
    snr_levels = (0:2:30);
    int8_scale = 128;
    tx_delay = 50;

    if(~endsWith(pwd(), 'RadioShift/src'))
        error("Please Run from Src in Radio Shift");
    end
    

    mod_types = categorical(["BPSK", "QPSK", "8PSK", ...
                                "16QAM","32QAM", "64QAM", "128QAM", "256QAM",...
                                "16APSK", "32APSK", "64APSK", "128APSK",...
                                 "FM", "AM-DSB-SC", "AM-SSB-SC"]); 
                        
    num_mod_types = length(mod_types);

    K_Factors = [4 0]; 


    set_ppm = (freq_scale_factor * MAX_PPM); % Convert to PPM
    data_directory = fullfile(Path, FolderName);
    fprintf("Data file directory is %s \n", string(data_directory));
    % Check if data files exist

    if exist(data_directory,'dir')
        fprintf("%s exists and generated signals will be saved here\n", string(data_directory));
    else
        fprintf("Creating directory at: %s\n", string(data_directory));
        [success,msg,msgID] = mkdir(data_directory);
        if ~(success)
            error(msgID,msg)
        end
    end
    

    channel = dlhdlhelperModClassTestChannel(...
        'SampleRate', fs, ...
        'SNR', snr_levels(1), ...
        'PathDelays', [0 1.8 3.4] / fs, ...
        'AveragePathGains', [0 -2 -10], ...
        'KFactor', K_Factors(channelType+1), ...
        'MaximumDopplerShift', 4, ...
        'MaximumClockOffset', set_ppm, ... %Set this as max PPM
        'HardSetOffsetPPM', hard_set_offset,... % If this is set as 1, we hardset the Max PPM, otherwise we use a random PPM
        'CenterFrequency', fc(1), ...
        'ChannelType', channelType ... % 0 = Rician Fading, 1 = Rayleigh Fading
        );

    disp(['Hardset PPM Offset: ', num2str(hard_set_offset)]);
    disp(['Maximum Clock Offset (PPM): ', num2str(set_ppm)]);
    
    %Setting up the Seed for Reproducibility
    rng(1235);

    channel_info = info(channel);
    disp(channel_info);
    total_frame_count = num_mod_types*frames_per_mod_type*length(snr_levels);
    fprintf("A total of %d frames will be generated...\n",total_frame_count);

    % all_IQ_int8 = zeros(spf, 2, total_frame_count, 'int8');
    all_IQ_float32 = zeros(spf, 2, total_frame_count, 'single');
    all_labels = zeros(1, total_frame_count, 'int64');
    all_SNRs = zeros(1, total_frame_count, 'int64');
    files_count_tracker = 1;
    tic;
    for mod = 1:num_mod_types
        elapsed_time = seconds(toc);
        elapsed_time.Format = 'hh:mm:ss';
        fprintf('%s - Generating %s frames\n', ...
        elapsed_time, mod_types(mod))

        label = string(mod_types(mod));
        label_idx = mod-1;
        
        disp(['Modulation: ', label, ' → Label Index: ', num2str(label_idx)]);

        dataSrc = dlhdlhelperModClassGetSource(mod_types(mod), sps, 2*spf, fs);
        modulator = dlhdlhelperModClassGetModulator(mod_types(mod), sps, fs);
        if contains(char(mod_types(mod)), {'FM','AM-DSB-SC','AM-SSB-SC'})
        % Analog modulation types use a center frequency of 100 MHz
        channel.CenterFrequency = 100e6;
        else
        % Digital modulation types use a center frequency of 902 MHz
        channel.CenterFrequency = 902e6;
        end
        wb = waitbar(0,sprintf("Generating frames for %s...", label));
        for j = (1:length(snr_levels))
            for p=(1:frames_per_mod_type)
                % Generate random data
                x = dataSrc();
                SNR = snr_levels(j);
                
                % Modulate
                y = modulator(x);
                channel.SNR = SNR;
                % Pass through independent channels
                reset(channel);
                rx_samples = channel(y);
                % rx_samples = y; % For Testing without Channel Effects
                % Remove transients from the beginning, trim to size, and normalize
                frame = dlhdlhelperModClassFrameGenerator(rx_samples, spf, spf, tx_delay, sps);
                
                % Save data file
                IQ = [real(frame),imag(frame)];
                % This has been commented out as we are no longer leveraging normalization in MATLAB
                    % IQ_Capped = IQ; % Create a copy for capping
                    % % Saving the Normalized and Scaled Data for the Int 8 Dataset
                    % IQ_Capped(IQ>5.5) = 5.5; % Capping values greater than 5.5
                    % IQ_Capped(IQ<-5.5) = -5.5; % Capping values less than -5.5
                    % IQ_Normalized = IQ_Capped/5.5; % Normalizing by 5.5 to keep most values in the -1 to 1 range
                    % frame_IQ = (IQ_Normalized * int8_scale);
                    % all_IQ_int8(:,:,files_count_tracker) = frame_IQ;
                % Saving the Raw IQ frames with no Scaling for Float32 Dataset
                frame_IQ = (IQ);
                all_IQ_float32(:,:,files_count_tracker) = frame_IQ;
                
                all_labels(files_count_tracker) = label_idx;
                all_SNRs(files_count_tracker) = SNR;

                files_count_tracker = files_count_tracker+1;
            end
            waitbar(j/length(snr_levels));
        end
        close(wb);
    end
    
    % all_IQ_int8 = permute(all_IQ_int8, [2, 1, 3]); % Convert to (frames, spf, 2)
    all_IQ_float32 = permute(all_IQ_float32, [2, 1, 3]); % Convert to (frames, spf, 2
    file_location = fullfile(data_directory,"MatGenData.h5");
    if isfile(file_location)
        delete(file_location);
    end

    % Saving Int8 Dataset as Int8 type
    % h5create(file_location, '/all_IQ_int8', [2, spf, total_frame_count],...
    % 'Datatype', 'int8');
    % h5write(file_location, '/all_IQ_int8', all_IQ_int8);
    % disp("Saving Int8 Dataset");

    % Saving Float32 Dataset as Single type which is Float32
    h5create(file_location, '/all_IQ_float32', [2, spf, total_frame_count],...
    'Datatype', 'single');
    h5write(file_location, '/all_IQ_float32', all_IQ_float32);
    disp("Saving Float32 Dataset");

    % Saving Labels Dataset
    h5create(file_location, '/all_labels', [1, total_frame_count],...
     'Datatype', 'int64');
    h5write(file_location, '/all_labels', all_labels);
    disp("Saving Labels Dataset");

    % Saving SNRs Dataset
    h5create(file_location, '/all_SNRs', [1, total_frame_count],...
     'Datatype', 'int64');
    h5write(file_location, '/all_SNRs', all_SNRs);
    disp("Saving SNRs Dataset");


end
