clear; clc; close all;
rng(42);

%% Параметры
Nr = 8;
Ni = 4;
SNR_dB = 10;
SIR_list = -10:2:10;
Nreal = 5000;

params_base = struct(...
    'Nr', Nr, 'Ni', Ni, 'K', 72, ...
    'SNR_dB', SNR_dB, 'SIR_dB', 0, ...
    'cci_type', 'QPSK', 'seed', 42);

fprintf('EM: канал + ICM\n');
fprintf('N_r=%d, N_i=%d, SNR=%d дБ, K=%d\n', Nr, Ni, SNR_dB, params_base.K);
fprintf('Данные: 16-QAM, DMRS: QPSK, CCI: QPSK\n');

%% Визуализация сетки
[~, info0] = generate_signal_model(params_base);
show_grid(info0);
fprintf('Позиции DMRS (1-индекс): %s\n', mat2str(info0.dmrs_idx));
fprintf('Число DMRS: %d,  Data+Control RE: %d\n\n', ...
    numel(info0.dmrs_idx), numel(info0.data_idx));

%% МОНТЕ-КАРЛО
nSIR = numel(SIR_list);
SINR_ideal_all = nan(Nreal, nSIR);
SINR_em_all    = nan(Nreal, nSIR);
SER_ideal_all  = nan(Nreal, nSIR);
SER_em_all     = nan(Nreal, nSIR);

for r = 1:Nreal
    params = params_base;
    params.seed = 1000 + r;

    for iSIR = 1:nSIR
        params.SIR_dB = SIR_list(iSIR);
        [Y, info] = generate_signal_model(params);

        % Идеальный IRC
        w_id = (info.h*info.h' + info.R_in_true) \ info.h;
        s_id = abs(w_id'*info.h)^2 / real(w_id'*info.R_in_true*w_id);
        SINR_ideal_all(r, iSIR) = 10*log10(s_id);
        SER_ideal_all(r, iSIR)  = compute_ser(w_id, Y, info);
        
        % EM
        [h_em, R_em] = em_channel_icm(Y, info.dmrs_idx, info.s_dmrs, ...
                                       info.const_data, info.M_data, ...
                                       1e-3, 20);
        w_em = (h_em*h_em' + R_em) \ h_em;
        s_em = abs(w_em'*info.h)^2 / real(w_em'*info.R_in_true*w_em);
        SINR_em_all(r, iSIR) = 10*log10(s_em);
        SER_em_all(r, iSIR)  = compute_ser(w_em, Y, info);
    end

    if mod(r, max(1,round(Nreal/10))) == 0
        fprintf('  r=%d/%d ', r, Nreal);
    end
end

%% Усреднение
SINR_ideal_m = mean(SINR_ideal_all, 1, 'omitnan');
SINR_em_m = mean(SINR_em_all, 1, 'omitnan');
SER_ideal_m = mean(SER_ideal_all, 1, 'omitnan');
SER_em_m = mean(SER_em_all, 1, 'omitnan');

fprintf('%-8s %14s %14s %10s\n', 'SIR (дБ)', 'SINR идеал', 'SINR EM', 'Δ (дБ)');
fprintf('%s\n', repmat('-', 1, 52));
for iSIR = 1:nSIR
    fprintf('%-8d %14.2f %14.2f %10.2f\n', ...
        SIR_list(iSIR), SINR_ideal_m(iSIR), SINR_em_m(iSIR), ...
        SINR_ideal_m(iSIR) - SINR_em_m(iSIR));
end

%% Графики
figure('Position',[100 100 1200 500], 'Name','EM vs Ideal IRC');

% SINR
subplot(1,2,1);
plot(SIR_list, SINR_ideal_m, 'k-', 'LineWidth', 2, ...
     'DisplayName','Идеальный IRC'); hold on;
plot(SIR_list, SINR_em_m, 'go-', 'LineWidth', 1.8, ...
     'MarkerSize', 7, 'DisplayName','EM');
grid on; grid minor;
xlabel('SIR (дБ)', 'FontSize', 12);
ylabel('SINR_{out} (дБ)', 'FontSize', 12);
title(sprintf('SINR_{out} от SIR (N_r=%d, N_i=%d, 16-QAM)', Nr, Ni), ...
      'FontSize', 12);
legend('Location','southeast', 'FontSize', 11);

% SER
subplot(1,2,2);
semilogy(SIR_list, max(SER_ideal_m, 1e-4), 'k-', 'LineWidth', 2, ...
         'DisplayName','Идеальный IRC'); hold on;
semilogy(SIR_list, max(SER_em_m, 1e-4), 'go-', 'LineWidth', 1.8, ...
         'MarkerSize', 7, 'DisplayName','EM');
grid on; grid minor;
xlabel('SIR (дБ)', 'FontSize', 12);
ylabel('SER (16-QAM)', 'FontSize', 12);
title('SER от SIR', 'FontSize', 12);
legend('Location','southwest', 'FontSize', 11);
ylim([1e-4 1]);

%% ГЕНЕРАТОР СИГНАЛА
function [Y, info] = generate_signal_model(params)
    if isfield(params,'seed') && ~isempty(params.seed)
        rng(params.seed);
    end

    Nr = params.Nr;  Ni = params.Ni;  K = params.K;
    SNR_dB = params.SNR_dB;  SIR_dB = params.SIR_dB;
    cci_type = 'QPSK';
    if isfield(params,'cci_type'), cci_type = params.cci_type; end

    N_sub = 12;  N_sym = 14;
    re_map = ones(N_sub, N_sym);

    re_map(:, 1) = 3;
    re_map(:, 2) = 3;
    re_map(1:2:end, 3) = 2;

    re_types = zeros(1, N_sub*N_sym);
    idx = 0;
    for s = 1:N_sym
        for f = 1:N_sub
            idx = idx + 1;
            re_types(idx) = re_map(f, s);
        end
    end
    re_types = re_types(1:K);

    dmrs_idx = find(re_types == 2);
    data_idx = find(re_types == 1 | re_types == 3);
    nP = numel(dmrs_idx);
    nD = numel(data_idx);

    % Созвездия
    const_pilot = (1/sqrt(2)) * [1+1i, -1+1i, 1-1i, -1-1i];
    M_pilot = 4;
    [I16, Q16] = meshgrid([-3 -1 1 3], [-3 -1 1 3]);
    const_data = (I16(:) + 1i*Q16(:)).' / sqrt(10);  
    M_data = 16;

    Ps = 1;
    Pi = Ps * 10^(-SIR_dB/10);
    Pn = (Ps/Nr) * 10^(-SNR_dB/10);

    h = sqrt(Ps/Nr)       * (randn(Nr,1)  + 1i*randn(Nr,1))  / sqrt(2);
    G = sqrt(Pi/(Ni*Nr))  * (randn(Nr,Ni) + 1i*randn(Nr,Ni)) / sqrt(2);

    s = zeros(1, K);
    s(data_idx) = const_data(randi(M_data, 1, nD));
    s(dmrs_idx) = const_pilot(randi(M_pilot, 1, nP));
    s_dmrs = s(dmrs_idx);

    switch cci_type
        case 'QPSK'
            I_sym = const_pilot(randi(M_pilot, Ni, K));
        case '16QAM'
            I_sym = const_data(randi(M_data, Ni, K));
        case 'Gaussian'
            I_sym = sqrt(1/2) * (randn(Ni,K) + 1i*randn(Ni,K));
    end

    N     = sqrt(Pn/2) * (randn(Nr,K) + 1i*randn(Nr,K));
    Y_sig = h * s;
    Y_int = G * I_sym;
    Y     = Y_sig + Y_int + N;

    R_in_true = G*G' + Pn*eye(Nr);

    info = struct();
    info.h = h;  info.G = G;
    info.s_true = s;  info.s_dmrs = s_dmrs;
    info.I_sym = I_sym;
    info.dmrs_idx = dmrs_idx;
    info.data_idx = data_idx;
    info.re_map = re_map;
    info.N_sub = N_sub;  info.N_sym = N_sym;
    info.K = K;  info.Nr = Nr;  info.Ni = Ni;
    info.const_data = const_data;  info.M_data = M_data;
    info.const_pilot = const_pilot;  info.M_pilot = M_pilot;
    info.R_in_true = R_in_true;
    info.Y_sig = Y_sig;  info.Y_int = Y_int;  info.N = N;
    info.Ps = Ps;  info.Pi = Pi;  info.Pn = Pn;
end

%% ВИЗУАЛИЗАЦИЯ СЕТКИ 
function show_grid(info)
    figure('Position',[100 100 800 500], 'Name','Resource grid (Fig. 1)');
    imagesc(info.re_map);

    cmap = [0.75 0.85 1.00;
            0.30 0.70 0.30;
            0.95 0.45 0.10];
    colormap(cmap);
    clim([1 3]);

    set(gca, 'YDir','normal', ...
             'XTick', 1:14, 'YTick', 1:12, ...
             'FontSize', 11);
    xlabel('Символ OFDM (время)', 'FontSize', 12);
    ylabel('Поднесущая (частота)', 'FontSize', 12);
    title(sprintf('Сетка ресурсов 12×14 — K=%d, |P|=%d', ...
                  info.K, numel(info.dmrs_idx)), 'FontSize', 12);

    hold on;
    % Используемая EM область: 6 символов × 12 поднесущих
    rectangle('Position', [0.5 0.5 6 12], ...
              'EdgeColor','k', 'LineWidth', 2, 'LineStyle','--');

    h_c = plot(NaN,NaN,'s', 'MarkerFaceColor',cmap(3,:), ...
               'MarkerEdgeColor','k', 'MarkerSize', 14);
    h_p = plot(NaN,NaN,'s', 'MarkerFaceColor',cmap(2,:), ...
               'MarkerEdgeColor','k', 'MarkerSize', 14);
    h_d = plot(NaN,NaN,'s', 'MarkerFaceColor',cmap(1,:), ...
               'MarkerEdgeColor','k', 'MarkerSize', 14);
    legend([h_c h_p h_d], ...
           {'Control','DMRS','Data'}, ...
           'Location','eastoutside', 'FontSize', 11);
    grid on;
end

%% SER на data/control RE
function ser = compute_ser(w, Y, info)
    n = 0;  err = 0;
    for k = info.data_idx
        shat = w' * Y(:, k);
        [~, mh] = min(abs(shat - info.const_data));
        [~, mt] = min(abs(info.s_true(k) - info.const_data));
        err = err + (mh ~= mt);
        n = n + 1;
    end
    ser = err / max(n, 1);
end

%% EM-АЛГОРИТМ 
function [h_em, R_in] = em_channel_icm(Y, dmrs_idx, s_dmrs, ...
                                        const_data, M_data, eps_em, maxIter)
    [Nr, K] = size(Y);
    nP = numel(dmrs_idx);

    Yp   = Y(:, dmrs_idx);
    h_ls = (Yp * s_dmrs') / (s_dmrs * s_dmrs');

    Rp = zeros(Nr, Nr);
    for i = 1:nP
        e = Yp(:,i) - h_ls * s_dmrs(i);
        Rp = Rp + e*e';
    end
    Rp = Rp / nP;

    Ryy = (Y*Y') / K;
    delta = max(real(min(eig(Ryy))), 1e-6);

    h_em = h_ls;
    R_in = Rp + delta * eye(Nr);

    const_row = const_data(:).';
    const_col = const_data(:);
    abs_c2    = abs(const_row).^2;
    conj_c    = conj(const_row);

    for iter = 1:maxIter
        % E-шаг
        RinvY  = R_in \ Y;
        yRinvy = real(sum(conj(Y).*RinvY, 1)).';
        hRinv  = (R_in \ h_em)';
        hRinvY = hRinv * Y;
        hRinvh = real(hRinv * h_em);

        cross = -2 * real( conj(const_row) .* hRinvY.' );
        quad  = hRinvh * abs_c2;
        Dist  = yRinvy + cross + quad;

        Dist  = Dist - min(Dist, [], 2); % Защита от переполнения exp()
        Ppost = exp(-Dist);
        Ppost = Ppost ./ sum(Ppost, 2);

        for i = 1:nP
            Ppost(dmrs_idx(i), :) = 0;
        end

        % M-шаг: h
        W_conj_c = Ppost * conj_c.';
        num = Y * W_conj_c;
        den = sum(Ppost * abs_c2.');

        for i = 1:nP
            k = dmrs_idx(i);
            num = num + conj(s_dmrs(i)) * Y(:, k);
            den = den + abs(s_dmrs(i))^2;
        end
        if den < 1e-12, den = 1e-12; end
        h_new = num / den;

        % M-шаг: Q
        Q = zeros(Nr, Nr);
        for m = 1:M_data
            w_m = Ppost(:, m).';
            E_m = Y - h_new * const_col(m);
            E_m = E_m .* sqrt(max(w_m, 0));
            Q = Q + E_m * E_m';
        end
        for i = 1:nP
            k = dmrs_idx(i);
            e = Y(:, k) - h_new * s_dmrs(i);
            Q = Q + e*e';
        end
        Q = Q / K;

        % Extended PPCA
        [Vq, Dq] = eig((Q + Q')/2);
        [lam, idx] = sort(real(diag(Dq)), 'descend');
        Vq  = Vq(:, idx);
        lam = max(lam, 0);

        Ni_hat = determine_Ni(lam, K, Nr);
        Lam_Ni = diag(lam(1:Ni_hat));
        V_Ni   = Vq(:, 1:Ni_hat);
        beta   = max((sum(lam) - sum(lam(1:Ni_hat))) / (Nr - Ni_hat), 1e-9);
        R_new  = V_Ni * (Lam_Ni - beta*eye(Ni_hat)) * V_Ni' + beta*eye(Nr);
        R_new  = (R_new + R_new') / 2;

        % сходимость
        dh = norm(h_new - h_em);
        dR = norm(R_new - R_in, 'fro');
        h_em = h_new;
        R_in = R_new;
        if dh < eps_em && dR < eps_em, break; end
    end
end

%% Algorithm 1: определение N_i (Minka BIC)
function Ni_hat = determine_Ni(lam, K, Nr)
    lam = lam(:);
    L = inf(Nr, 1);
    for p = 1:Nr
        if p < Nr
            beta_p = max((sum(lam) - sum(lam(1:p))) / (Nr - p), 1e-12);
        else
            beta_p = 1e-12;
        end
        m = Nr*p - p*(p+1)/2;
        L(p) = sum(log(lam(1:p) + 1e-12)) + (Nr-p)*log(beta_p) ...
               + (m+p)/K * log(K);
    end
    [~, Ni_hat] = min(L);
end