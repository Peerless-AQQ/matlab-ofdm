clear; close all; clc;

%% ПАРАМЕТРЫ 
Nused  = 1320;
Sf     = 6;
Fs     = 30.72e6;
Delta_f = 15e3;

% Comb-массив с расширением
N_ext = 6;
Ncomb_ext = Nused + 2*N_ext;     % = 1332 = 6*222
Ng_local = round(144 * Ncomb_ext / 2048);

used_start = N_ext + 1;          % = 7
used_end   = N_ext + Nused;      % = 1326

pilot_in_used = (1:Sf:Nused).';
Npilot        = length(pilot_in_used);
pilot_in_ext  = pilot_in_used + N_ext;
data_in_used  = setdiff((1:Nused).', pilot_in_used);
data_in_ext   = data_in_used + N_ext;

% Веса линейной интерполяции
n_vec   = (1:Nused).';
i_left  = floor((n_vec - 1) / Sf) + 1;
i_left  = min(i_left,  Npilot);
i_right = min(i_left + 1, Npilot);
pos_l   = pilot_in_used(i_left);
pos_r   = pilot_in_used(i_right);
denom   = pos_r - pos_l;
denom(denom == 0) = 1;
t_vec   = (n_vec - pos_l) ./ denom;
t_vec(pos_l == pos_r) = 0;
w_r     = t_vec;
w_l     = 1 - t_vec;

% Канал
v_kmh = 30; fc = 4e9;
c = physconst('lightspeed');
fd = (v_kmh*1000/3600)/c*fc;
delay_spread_ns = 100;

tdl = nrTDLChannel;
tdl.DelayProfile        = 'TDL-E';
tdl.DelaySpread         = delay_spread_ns * 1e-9;
tdl.MaximumDopplerShift = fd;
tdl.SampleRate          = Fs;
tdl.NumTransmitAntennas = 1;
tdl.NumReceiveAntennas  = 1;
tdl.NormalizePathGains  = true;

info_tdl = info(tdl);
max_delay_s = max(info_tdl.PathDelays);

Fs_comb = Ncomb_ext * Delta_f;
max_delay_comb_samples = ceil(max_delay_s * Fs_comb);
L_trunc = max_delay_comb_samples + 16;

fprintf('=== Визуализация |S(ω)| и BER, TDL-E ===\n');
fprintf('Nused=%d, Sf=%d, Ncomb_ext=%d, L_trunc=%d\n\n', ...
        Nused, Sf, Ncomb_ext, L_trunc);

M = 16; Mod_Order = 4;
K_dB = 22;

%% ТЕОРЕТИЧЕСКИЙ BER 
function ber = theoretical_ber_tdl_e(EbN0_dB, M, K_dB)
    k = log2(M); L = sqrt(M);
    EbN0 = 10.^(EbN0_dB/10);
    ber = (2/k) * (1 - 1/L) * erfc(sqrt(3 * EbN0 * k / (2 * (M - 1))));
    K = 10^(K_dB/10);
    ber = ber / (1 + 1/K);
end

% BER-кривые

SNRdB = 0:5:30;
nSNR  = length(SNRdB);
BER_comb = zeros(size(SNRdB));
BER_ls   = zeros(size(SNRdB));

MIN_ERRORS   = 100;
MAX_BITS_CAP = 5e7;
rng(42);

BER_pred = theoretical_ber_tdl_e(SNRdB - 10*log10(Mod_Order), M, K_dB);

for i_snr = 1:nSNR
    EsN0_lin = 10^(SNRdB(i_snr)/10);
    err_comb = 0; err_ls = 0; total_bits = 0;

    fprintf('BER SNR = %5.1f дБ: ', SNRdB(i_snr));
    tic;

    while (err_comb < MIN_ERRORS || err_ls < MIN_ERRORS) ...
          && total_bits < MAX_BITS_CAP

        data_bits = randi([0 1], length(data_in_used) * Mod_Order, 1);
        data_syms = qammod(data_bits, M, 'InputType', 'bit', ...
                           'UnitAveragePower', true);
        pilot_bits = randi([0 1], Npilot * Mod_Order, 1);
        pilot_syms = qammod(pilot_bits, M, 'InputType', 'bit', ...
                            'UnitAveragePower', true);

        freq_syms = zeros(Ncomb_ext, 1);
        freq_syms(data_in_ext)  = data_syms;
        freq_syms(pilot_in_ext) = pilot_syms;

        x_time = ifft(freq_syms) * sqrt(Ncomb_ext);
        x_cp   = [x_time(end-Ng_local+1:end); x_time];

        reset(tdl);
        y_ch = tdl(complex(x_cp));
        noise = sqrt(1/(2*EsN0_lin)) * ...
                (randn(size(y_ch)) + 1j*randn(size(y_ch)));
        y_rx = y_ch + noise;

        y_time = y_rx(Ng_local+1:end);
        Y_nat  = fft(y_time) / sqrt(Ncomb_ext);

        pilot_syms_nat = freq_syms(pilot_in_ext);
        H_LS_pil = Y_nat(pilot_in_ext) ./ pilot_syms_nat;

        if any(~isfinite(H_LS_pil))
            continue;
        end

        % Comb-DFT
        H_comb = zeros(Ncomb_ext, 1);
        H_comb(pilot_in_ext) = H_LS_pil;
        slope_L = H_LS_pil(2) - H_LS_pil(1);
        H_comb(1) = H_LS_pil(1) - slope_L;
        slope_R = H_LS_pil(end) - H_LS_pil(end-1);
        H_comb(Ncomb_ext - 5) = H_LS_pil(end) + slope_R;

        h_comb = ifft(H_comb) * sqrt(Ncomb_ext);
        h_trunc = zeros(Ncomb_ext, 1);
        h_trunc(1:L_trunc) = h_comb(1:L_trunc) * Sf;
        H_est_full = fft(h_trunc) / sqrt(Ncomb_ext);
        H_est_comb = H_est_full(used_start:used_end);

        % LS + linear
        H_est_ls = w_l .* H_LS_pil(i_left) + w_r .* H_LS_pil(i_right);

        % Эквалайзер
        Y_data_nat  = Y_nat(data_in_ext);
        tx_syms_nat = freq_syms(data_in_ext);
        tx_bits_used = qamdemod(tx_syms_nat, M, 'OutputType', 'bit', ...
                                'UnitAveragePower', true);

        H_d = H_est_comb(data_in_used);
        X_eq = conj(H_d) .* Y_data_nat ./ (abs(H_d).^2 + 1/EsN0_lin + eps);
        X_eq(~isfinite(X_eq)) = 0;
        rx = qamdemod(X_eq, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_comb = err_comb + sum(tx_bits_used ~= rx);

        H_d = H_est_ls(data_in_used);
        X_eq = conj(H_d) .* Y_data_nat ./ (abs(H_d).^2 + 1/EsN0_lin + eps);
        X_eq(~isfinite(X_eq)) = 0;
        rx = qamdemod(X_eq, M, 'OutputType', 'bit', 'UnitAveragePower', true);
        err_ls = err_ls + sum(tx_bits_used ~= rx);

        total_bits = total_bits + length(tx_bits_used);
    end

    BER_comb(i_snr) = err_comb / total_bits;
    BER_ls(i_snr)   = err_ls   / total_bits;

    fprintf('t=%.1f с | Comb: %.3e | LS: %.3e', ...
            toc, BER_comb(i_snr), BER_ls(i_snr));
    if err_ls > 0 && err_comb > 0
        fprintf(' | Comb vs LS: %+.2f дБ\n', ...
                10*log10(BER_ls(i_snr)/BER_comb(i_snr)));
    else
        fprintf('\n');
    end
end

% График BER
figure('Position', [60 60 800 560]);
semilogy(SNRdB, BER_pred, 'k-', 'LineWidth', 2.5, ...
    'DisplayName', 'Теория');
hold on;
valid_ls   = BER_ls   > 0;
valid_comb = BER_comb > 0;
semilogy(SNRdB(valid_ls), BER_ls(valid_ls), 'b-o', 'LineWidth', 1.8, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b', ...
    'DisplayName', 'LS + линейная интерполяция');
semilogy(SNRdB(valid_comb), BER_comb(valid_comb), 'r-s', 'LineWidth', 1.8, ...
    'MarkerSize', 8, 'MarkerFaceColor', 'r', ...
    'DisplayName', 'Comb-DFT');
grid on;
xlabel('SNR (дБ)', 'FontSize', 12);
ylabel('BER', 'FontSize', 12);
title(sprintf('BER: Comb-DFT vs LS, TDL-E (DS=%d нс)', delay_spread_ns), ...
      'FontSize', 13);
legend('Location', 'southwest', 'FontSize', 11);
axis([min(SNRdB) max(SNRdB) 1e-6 1]);

% Один OFDM-символ при SNR = 20 дБ
SNRdB_plot = 20;
EsN0_plot = 10^(SNRdB_plot/10);
rng(7);

data_bits = randi([0 1], length(data_in_used) * Mod_Order, 1);
data_syms = qammod(data_bits, M, 'InputType', 'bit', ...
                   'UnitAveragePower', true);
pilot_bits = randi([0 1], Npilot * Mod_Order, 1);
pilot_syms = qammod(pilot_bits, M, 'InputType', 'bit', ...
                    'UnitAveragePower', true);

freq_syms = zeros(Ncomb_ext, 1);
freq_syms(data_in_ext)  = data_syms;
freq_syms(pilot_in_ext) = pilot_syms;

x_time = ifft(freq_syms) * sqrt(Ncomb_ext);
x_cp   = [x_time(end-Ng_local+1:end); x_time];

reset(tdl);
y_ch = tdl(complex(x_cp));
noise = sqrt(1/(2*EsN0_plot)) * ...
        (randn(size(y_ch)) + 1j*randn(size(y_ch)));
y_rx = y_ch + noise;

y_time = y_rx(Ng_local+1:end);
Y_nat  = fft(y_time) / sqrt(Ncomb_ext);

pilot_syms_nat = freq_syms(pilot_in_ext);
H_LS_pil = Y_nat(pilot_in_ext) ./ pilot_syms_nat;

% Comb-DFT
H_comb = zeros(Ncomb_ext, 1);
H_comb(pilot_in_ext) = H_LS_pil;
slope_L = H_LS_pil(2) - H_LS_pil(1);
H_comb(1) = H_LS_pil(1) - slope_L;
slope_R = H_LS_pil(end) - H_LS_pil(end-1);
H_comb(Ncomb_ext - 5) = H_LS_pil(end) + slope_R;

h_comb = ifft(H_comb) * sqrt(Ncomb_ext);
h_trunc = zeros(Ncomb_ext, 1);
h_trunc(1:L_trunc) = h_comb(1:L_trunc) * Sf;
H_est_full = fft(h_trunc) / sqrt(Ncomb_ext);
H_est_comb = H_est_full(used_start:used_end);

% LS + linear
H_est_ls = w_l .* H_LS_pil(i_left) + w_r .* H_LS_pil(i_right);

% Истинный канал через единичный OFDM-символ
reset(tdl);
X_id = zeros(Ncomb_ext, 1);
X_id(used_start:used_end) = 1;
x_id_t = ifft(X_id) * sqrt(Ncomb_ext);
x_id_cp = [x_id_t(end-Ng_local+1:end); x_id_t];
y_id = tdl(complex(x_id_cp));
H_true_full = fft(y_id(Ng_local+1:end)) / sqrt(Ncomb_ext);
H_true_used = H_true_full(used_start:used_end);

freq_MHz = [(-660:-1), (1:660)] * Delta_f / 1e6;

figure('Position', [80 80 1100 700]);

subplot(2,1,1);
plot(freq_MHz, 20*log10(abs(H_true_used)), 'k-', 'LineWidth', 1.6); hold on;
plot(freq_MHz, 20*log10(abs(H_est_ls)), 'b--', 'LineWidth', 1.2);
plot(freq_MHz, 20*log10(abs(H_est_comb)), 'r-', 'LineWidth', 1.3);
grid on;
xlabel('Частота, МГц', 'FontSize', 11);
ylabel('|S(\omega)|, дБ', 'FontSize', 11);
title(sprintf('Спектральная плотность |S(\\omega)| после интерполяции (SNR = %d дБ)', ...
              SNRdB_plot), 'FontSize', 12);
legend({'|S_{true}(\omega)|', '|S_{LS}(\omega)|', '|S_{comb}(\omega)|'}, ...
       'Location', 'best', 'FontSize', 10);
xlim([min(freq_MHz) max(freq_MHz)]);

subplot(2,1,2);
plot(freq_MHz, 20*log10(abs(H_true_used)), 'k-', 'LineWidth', 1.6); hold on;
plot(freq_MHz, 20*log10(abs(H_est_ls)), 'b--', 'LineWidth', 1.2);
plot(freq_MHz, 20*log10(abs(H_est_comb)), 'r-', 'LineWidth', 1.3);
grid on;
xlabel('Частота, МГц', 'FontSize', 11);
ylabel('|S(\omega)|, дБ', 'FontSize', 11);
title('Центральная часть спектра (\pm 3 МГц)', 'FontSize', 12);
legend({'|S_{true}(\omega)|', '|S_{LS}(\omega)|', '|S_{comb}(\omega)|'}, ...
       'Location', 'best', 'FontSize', 10);
xlim([-3 3]);