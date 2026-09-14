function summaryTable = run_sweep_bw_mod(bwList, modList)
%RUN_SWEEP_BW_MOD Plan d'expérience croisé largeur de bande x ordre de modulation.
%
%   Rejoue les neuf scénarios S0 à S8 pour chaque combinaison (bande,
%   modulation) et écrit results/sweep_bw_mod/sweep_bande_modulation.csv.
%
%   Le croisement, plutôt que deux balayages indépendants, permet de vérifier
%   que les deux facteurs n'interagissent pas : c'est ce qui autorise à
%   conclure que la largeur de bande gouverne la difficulté de linéarisation
%   tandis que l'ordre de modulation ne resserre que l'exigence normative.
%
%   Usage :
%       run_sweep_bw_mod                       % 30/40/50 MHz x QPSK..256QAM
%       run_sweep_bw_mod([30 50], {'64QAM'})   % sous-ensemble
%
%   Le nombre de sous-porteuses actives suit la bande, de sorte que
%   l'occupation spectrale reste proportionnée : 250 à 30 MHz, 332 à 40 MHz
%   et 416 à 50 MHz, pour un espacement de 120 kHz.
%
%   ATTENTION — le scénario S7 impose une porteuse 5G NR. Le générateur du
%   5G Toolbox n'accepte pas toutes les combinaisons (bande, numérologie) :
%   hors des configurations valides au sens de TS 38.104, il échoue et la
%   chaîne bascule sur le générateur OFDM/QAM interne. La colonne Waveform
%   du CSV enregistre la forme d'onde RÉELLEMENT employée, et c'est elle qui
%   fait foi lors de l'interprétation, non l'intention du scénario.

if nargin < 1 || isempty(bwList),  bwList  = [30 40 50]; end
if nargin < 2 || isempty(modList), modList = {'QPSK','16QAM','64QAM','256QAM'}; end

% Sous-porteuses actives par largeur de bande (espacement de 120 kHz).
nscMap = containers.Map([30 40 50], [250 332 416]);
modOrder = containers.Map({'QPSK','16QAM','64QAM','256QAM'}, {4, 16, 64, 256});

scenarios = {'S0','S1','S2','S3','S4','S5','S6','S7','S8'};

base = defaultConfig();
base.toolboxes = toolboxCheck();
outDir = fullfile(base.paths.root, 'results', 'sweep_bw_mod');
if ~exist(outDir, 'dir'), mkdir(outDir); end

summaryTable = table();
nTot = numel(bwList) * numel(modList) * numel(scenarios);
k = 0;
ticAll = tic;

for bw = bwList
    for im = 1:numel(modList)
        modName = modList{im};
        M = modOrder(modName);
        for is = 1:numel(scenarios)
            sc = scenarios{is};
            k = k + 1;
            fprintf('[%3d/%3d] %3d MHz | %-6s | %s ... ', k, nTot, bw, modName, sc);

            cfg = defaultConfig();
            cfg.toolboxes = toolboxCheck();
            cfg = scenarioConfig(cfg, sc);

            % Le balayage prime sur la configuration propre au scénario :
            % appliqué APRÈS scenarioConfig, il écrase les valeurs que S7
            % fixe pour lui-même.
            cfg.bandwidth    = bw * 1e6;
            cfg.nSubcarriers = nscMap(bw);
            cfg.modulation   = modName;
            cfg.qamM         = M;
            cfg.plots.visible = 'off';
            cfg.plots.enabled = false;

            try
                res = runDPDScenario(cfg);
                T = res.summary;
                T.BW_config_MHz = repmat(bw,     height(T), 1);
                T.NSC           = repmat(nscMap(bw), height(T), 1);
                T.ModOrder      = repmat(M,      height(T), 1);
                T.ModName       = repmat(string(modName), height(T), 1);
                summaryTable = [summaryTable; T]; %#ok<AGROW>
                fprintf('ok (%.1f s)\n', sum(T.Runtime_s));
            catch ME
                fprintf('ECHEC : %s\n', ME.message);
            end
        end
    end
end

csvFile = fullfile(outDir, 'sweep_bande_modulation.csv');
writetable(summaryTable, csvFile);
fprintf('\n%d lignes écrites dans %s (%.1f min)\n', ...
        height(summaryTable), csvFile, toc(ticAll)/60);
end
