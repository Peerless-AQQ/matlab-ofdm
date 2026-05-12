clear; close all; clc;

%% 1. Параметры системы
Nfft = 1024;                % Размер FFT
Ng = round(Nfft / 4);       % Длина CP (256)
Nused = 840;                % Используемые поднесущие

Mod_Order = 4;              % 4 = 16-QAM
M = 2^Mod_Order;            % Порядок модуляции

Fs = 20e6;                  % Частота дискретизации 20 MHz
Ts = 1/Fs;                  % Период дискретизации

% Параметры канала
channel = comm.RayleighChannel(...
    'SampleRate', Fs, ...
    'PathDelays', [0 110 190 410] * 1e-9, ...
    'AveragePathGains', [0 -9.7 -19.2 -22.8], ...
    'NormalizePathGains', true, ...
    'MaximumDopplerShift', 5, ...
    'DopplerSpectrum', doppler('Jakes'), ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', 42);

% Параметры пилотов
pilot_spacing = 4;
pilot_indices = 1:pilot_spacing:Nused;
data_indices = setdiff(1:Nused, pilot_indices);

% SNR параметры
SNRdB = [0:5:30, 40];
BER = zeros(size(SNRdB));

rng(42);        % Фиксированный seed для воспроизводимости

fprintf('========================================\n');
fprintf('OFDM with MATLAB Rayleigh Channel\n');
fprintf('Modulation: %d-QAM\n', M);
fprintf('Sample rate: %.2f MHz\n', Fs/1e6);
fprintf('FFT size: %d, CP length: %d\n', Nfft, Ng);
fprintf('Used subcarriers: %d, Pilots: %d, Data: %d\n', Nused, length(pilot_indices), length(data_indices));
fprintf('Pilot spacing: %d\n', pilot_spacing);
fprintf('========================================\n\n');

%% Основной цикл по SNR
for i_snr = 1:length(SNRdB)
    
    % Расчет шума
    SNR_linear = 10^(SNRdB(i_snr)/10);
    signal_power = 1;
    noise_var = signal_power / SNR_linear;
    
    err_bits = 0;
    total_bits = 0;
    iter = 0;
    max_iter = 100;
    Target_Errors = 500;
    
    Constellation = qammod(0:M-1, M, 'UnitAveragePower', true);
    
    save_frame = (i_snr == length(SNRdB));
    
    if save_frame
        all_H_true = [];
        all_H_est = [];
    end
    
    while (iter < max_iter) && (err_bits < Target_Errors)
        iter = iter + 1;
        
        %% ПЕРЕДАТЧИК
        data_bits = randi([0 1], Nused * Mod_Order, 1);
        total_bits = total_bits + length(data_bits);
        
        mod_data = qammod(data_bits, M, 'InputType', 'bit', 'UnitAveragePower', true);
        
        X = zeros(Nfft, 1);
        used_indices_left = 2:(Nused/2 + 1);
        used_indices_right = Nfft - (Nused/2) + 1 : Nfft;
        used_indices = [used_indices_left, used_indices_right];
        X(used_indices) = mod_data;
        
        x_time = ifft(ifftshift(X)) * sqrt(Nfft);
        x_cp = [x_time(end-Ng+1:end); x_time];
        
        %% КАНАЛ
        reset(channel);
        x_cp_complex = complex(x_cp);
        y_channel = channel(x_cp_complex);
        
        %% ШУМ
        noise = sqrt(noise_var/2) * (randn(size(y_channel)) + 1j*randn(size(y_channel)));
        y_rx = y_channel + noise;
        
        %% ПРИЁМНИК
        y_time = y_rx(Ng+1:end);
        Y_freq = fftshift(fft(y_time)) / sqrt(Nfft);
        Y_data = Y_freq(used_indices);
        
        %% ОЦЕНКА КАНАЛА
        % LS оценка на пилотах
        H_LS_pilots = Y_data(pilot_indices) ./ mod_data(pilot_indices);
        
        try
            H_est = interp1(pilot_indices, H_LS_pilots, 1:Nused, 'spline');
        catch
            H_est = interp1(pilot_indices, H_LS_pilots, 1:Nused, 'pchip');
        end
        H_est = H_est(:);
        
        window_size = 5;
        H_est_mag = medfilt1(abs(H_est), window_size);
        H_est_phase = medfilt1(angle(H_est), window_size);
        H_est = H_est_mag .* exp(1j * H_est_phase);
        
        %% ДЕМОДУЛЯЦИЯ
        H_power = abs(H_est).^2;
        X_eq = conj(H_est) .* Y_data ./ (H_power + noise_var);
        
        rx_symbols = X_eq(data_indices);
        rx_bits_data = qamdemod(rx_symbols, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        
        bit_mask = true(Nused * Mod_Order, 1);
        for p = pilot_indices
            bit_mask((p-1)*Mod_Order + 1 : p*Mod_Order) = false;
        end
        tx_bits_data = data_bits(bit_mask);
        
        min_len = min(length(tx_bits_data), length(rx_bits_data));
        err_bits = err_bits + sum(tx_bits_data(1:min_len) ~= rx_bits_data(1:min_len));
        
        %% СОХРАНЕНИЕ ДАННЫХ ДЛЯ ВИЗУАЛИЗАЦИИ
        if save_frame && iter == 1
            original_mod_data = mod_data;
            original_X = X;
            recovered_mod_data = zeros(size(mod_data));
            recovered_mod_data(data_indices) = rx_symbols;
            recovered_mod_data(pilot_indices) = mod_data(pilot_indices);
            
            X_recovered = zeros(Nfft, 1);
            X_recovered(used_indices) = recovered_mod_data;
            
            Y_received = zeros(Nfft, 1);
            Y_received(used_indices) = Y_data;
            
            H_true_approx = interp1(pilot_indices, H_LS_pilots, 1:Nused, 'spline');
            H_true_data = H_true_approx(:);
            
            original_bits_data = data_bits(bit_mask);
            recovered_bits_data = rx_bits_data(1:min_len);
            
            H_estimated = H_est;
            H_LS_pilots_saved = H_LS_pilots;
            
            all_H_true = H_true_data;
            all_H_est = H_estimated;
        end
    end
    
    BER(i_snr) = err_bits / (total_bits * (Nused - length(pilot_indices)) / Nused);
    fprintf('SNR = %2d dB: BER = %.6f (Errors: %d, Bits: %.0f)\n', ...
        SNRdB(i_snr), BER(i_snr), err_bits, total_bits * (Nused - length(pilot_indices)) / Nused);
end

%% ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ

% BER график
figure('Position', [100, 100, 800, 600]);
semilogy(SNRdB, BER, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
title('BER в зависимости от отношения сигнал/шум', 'FontSize', 14);
grid on;
axis([min(SNRdB) max(SNRdB) 1e-6 1]);

% Теоретическая кривая для Rayleigh канала
if exist('berfading', 'file')
    EbN0_theory = SNRdB - 10*log10(Mod_Order);
    BER_theory = berfading(EbN0_theory, 'qam', M, 1);
    hold on;
    semilogy(SNRdB, BER_theory, 'r--', 'LineWidth', 1.5);
    legend('Моделирование (LS оценка)', 'Theoretical Rayleigh', 'Location', 'southwest');
end

%% ОЦЕНКА КАНАЛА: АМПЛИТУДА
figure('Position', [200, 200, 800, 500]);
subplot(2,1,1);
plot(1:Nused, abs(H_true_data), 'b-', 'LineWidth', 1.5); hold on;
plot(1:Nused, abs(H_estimated), 'r--', 'LineWidth', 1.2);
plot(pilot_indices, abs(H_LS_pilots_saved), 'go', 'MarkerSize', 4, 'MarkerFaceColor', 'g');
xlabel('Индекс поднесущей', 'FontSize', 10);
ylabel('|H|', 'FontSize', 10);
title('Влияние канала (истиное значение и оценка LS)', 'FontSize', 12);
legend('True (approx)', 'LS Estimate', 'Pilot LS', 'Location', 'best');
grid on;

subplot(2,1,2);
plot(1:Nused, angle(H_true_data), 'b-', 'LineWidth', 1.5); hold on;
plot(1:Nused, angle(H_estimated), 'r--', 'LineWidth', 1.2);
xlabel('Subcarrier Index', 'FontSize', 10);
ylabel('Phase (rad)', 'FontSize', 10);
title('Влияние канала (истиное значение и оценка LS)', 'FontSize', 12);
legend('True (approx)', 'LS Estimate', 'Location', 'best');
grid on;

%% СПЕКТРЫ В ЧАСТОТНОЙ ОБЛАСТИ
figure('Position', [100, 100, 1200, 600]);
freq_axis = (-Nfft/2:Nfft/2-1) / Nfft;

figure(4);
stem(freq_axis, abs(fftshift(original_X)), 'b-', 'LineWidth', 1);
xlabel('Normalized Frequency', 'FontSize', 10);
ylabel('|X(f)|', 'FontSize', 10);
title('Оригинальный спектр', 'FontSize', 12);
xlim([-0.5 0.5]); grid on;

figure(5);
stem(freq_axis, abs(fftshift(Y_received)), 'r-', 'LineWidth', 1);
xlabel('Normalized Frequency', 'FontSize', 10);
ylabel('|Y(f)|', 'FontSize', 10);
title('Принятый спектр (с шумом)', 'FontSize', 12);
xlim([-0.5 0.5]); grid on;

figure(6);
stem(freq_axis, abs(fftshift(original_X)), 'b-', 'LineWidth', 1); hold on;
stem(freq_axis, abs(fftshift(X_recovered)), 'g--', 'LineWidth', 1);
xlabel('Normalized Frequency', 'FontSize', 10);
ylabel('|X(f)|', 'FontSize', 10);
title('Оригинальный vs восстановленный', 'FontSize', 12);
legend('Original', 'Recovered', 'Location', 'best');
xlim([-0.5 0.5]); grid on;

%% СОЗВЕЗДИЕ
figure(3);
plot(real(original_mod_data), imag(original_mod_data), 'b.', 'MarkerSize', 6); hold on;
plot(real(recovered_mod_data(data_indices)), imag(recovered_mod_data(data_indices)), 'r.', 'MarkerSize', 4);
xlabel('In-Phase', 'FontSize', 12);
ylabel('Quadrature', 'FontSize', 12);
title('Созвездие: первоначальное и восстановленное', 'FontSize', 14);
legend('Original', 'Recovered', 'Location', 'best');
axis([-1.5 1.5 -1.5 1.5]);
axis square; grid on;
