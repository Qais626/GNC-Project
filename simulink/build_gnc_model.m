function build_gnc_model()
%BUILD_GNC_MODEL Creates a complete GNC Simulink model
%
%   BUILD_GNC_MODEL() creates a Simulink model for spacecraft attitude
%   control demonstrating PD controller with reaction wheel actuators.
%
%   Author: GNC Engineering Team
%   Date: 2026

    clc;
    fprintf('========================================\n');
    fprintf('  GNC Simulink Model Builder\n');
    fprintf('  KSC-Moon-Jupiter Mission\n');
    fprintf('========================================\n\n');

    modelName = 'GNC_System';

    % Close and delete existing
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
    if exist([modelName '.slx'], 'file')
        delete([modelName '.slx']);
    end

    % Create new model
    fprintf('Creating Simulink model: %s.slx\n', modelName);
    new_system(modelName);
    open_system(modelName);

    % Configure solver
    set_param(modelName, 'Solver', 'ode4', 'FixedStep', '0.01', 'StopTime', '100');

    fprintf('Building model components...\n');

    %% ===== ATTITUDE DYNAMICS SUBSYSTEM =====
    fprintf('  [1/4] Attitude Dynamics\n');

    dynSys = [modelName '/Attitude_Dynamics'];
    add_block('simulink/Ports & Subsystems/Subsystem', dynSys, ...
              'Position', [200 150 300 230]);
    set_param(dynSys, 'BackgroundColor', 'lightBlue');
    Simulink.SubSystem.deleteContents(dynSys);

    % Ports
    add_block('simulink/Sources/In1', [dynSys '/Torque'], 'Position', [30 100 60 120]);
    add_block('simulink/Sinks/Out1', [dynSys '/Theta'], 'Position', [500 70 530 90]);
    add_block('simulink/Sinks/Out1', [dynSys '/Omega'], 'Position', [500 130 530 150]);

    % Dynamics: theta_ddot = tau / I
    % Single axis for simplicity (can extend to 3-axis)
    add_block('simulink/Math Operations/Gain', [dynSys '/Inv_I'], ...
              'Gain', '1/Ixx', 'Position', [150 95 200 125]);

    add_block('simulink/Continuous/Integrator', [dynSys '/Omega_Int'], ...
              'InitialCondition', 'omega0_x', 'Position', [270 95 310 125]);

    add_block('simulink/Continuous/Integrator', [dynSys '/Theta_Int'], ...
              'InitialCondition', 'theta0_x', 'Position', [380 95 420 125]);

    add_line(dynSys, 'Torque/1', 'Inv_I/1');
    add_line(dynSys, 'Inv_I/1', 'Omega_Int/1');
    add_line(dynSys, 'Omega_Int/1', 'Theta_Int/1');
    add_line(dynSys, 'Theta_Int/1', 'Theta/1');
    add_line(dynSys, 'Omega_Int/1', 'Omega/1');

    %% ===== CONTROLLER SUBSYSTEM =====
    fprintf('  [2/4] PD Controller\n');

    ctrlSys = [modelName '/PD_Controller'];
    add_block('simulink/Ports & Subsystems/Subsystem', ctrlSys, ...
              'Position', [200 280 300 360]);
    set_param(ctrlSys, 'BackgroundColor', 'orange');
    Simulink.SubSystem.deleteContents(ctrlSys);

    % Ports
    add_block('simulink/Sources/In1', [ctrlSys '/Theta_Cmd'], 'Position', [30 30 60 50]);
    add_block('simulink/Sources/In1', [ctrlSys '/Theta'], 'Position', [30 80 60 100]);
    add_block('simulink/Sources/In1', [ctrlSys '/Omega'], 'Position', [30 130 60 150]);
    add_block('simulink/Sinks/Out1', [ctrlSys '/Tau'], 'Position', [450 80 480 100]);

    % Error computation
    add_block('simulink/Math Operations/Sum', [ctrlSys '/Theta_Err'], ...
              'Inputs', '+-', 'Position', [120 50 150 80]);

    % PD gains (positive gains for stability)
    add_block('simulink/Math Operations/Gain', [ctrlSys '/Kp'], ...
              'Gain', 'Kp_x', 'Position', [200 45 250 75]);
    add_block('simulink/Math Operations/Gain', [ctrlSys '/Kd'], ...
              'Gain', 'Kd_x', 'Position', [200 105 250 135]);

    % Sum PD (Kp*error - Kd*omega for damping)
    add_block('simulink/Math Operations/Sum', [ctrlSys '/PD_Sum'], ...
              'Inputs', '+-', 'Position', [310 70 340 100]);

    % Saturation
    add_block('simulink/Discontinuities/Saturation', [ctrlSys '/Sat'], ...
              'UpperLimit', 'tau_max', 'LowerLimit', '-tau_max', ...
              'Position', [380 70 410 100]);

    add_line(ctrlSys, 'Theta_Cmd/1', 'Theta_Err/1');
    add_line(ctrlSys, 'Theta/1', 'Theta_Err/2');
    add_line(ctrlSys, 'Theta_Err/1', 'Kp/1');
    add_line(ctrlSys, 'Omega/1', 'Kd/1');
    add_line(ctrlSys, 'Kp/1', 'PD_Sum/1');
    add_line(ctrlSys, 'Kd/1', 'PD_Sum/2');
    add_line(ctrlSys, 'PD_Sum/1', 'Sat/1');
    add_line(ctrlSys, 'Sat/1', 'Tau/1');

    %% ===== ACTUATOR SUBSYSTEM =====
    fprintf('  [3/4] Reaction Wheel\n');

    actSys = [modelName '/Reaction_Wheel'];
    add_block('simulink/Ports & Subsystems/Subsystem', actSys, ...
              'Position', [200 410 300 470]);
    set_param(actSys, 'BackgroundColor', 'gray');
    Simulink.SubSystem.deleteContents(actSys);

    add_block('simulink/Sources/In1', [actSys '/Tau_Cmd'], 'Position', [30 50 60 70]);
    add_block('simulink/Sinks/Out1', [actSys '/Tau_Out'], 'Position', [350 35 380 55]);
    add_block('simulink/Sinks/Out1', [actSys '/Momentum'], 'Position', [350 75 380 95]);

    % First-order dynamics
    add_block('simulink/Continuous/Transfer Fcn', [actSys '/RW_TF'], ...
              'Numerator', '[1]', 'Denominator', '[rw_tau, 1]', ...
              'Position', [150 40 230 70]);

    % Momentum integration
    add_block('simulink/Continuous/Integrator', [actSys '/H_Int'], ...
              'InitialCondition', '0', 'Position', [270 70 300 100]);

    add_line(actSys, 'Tau_Cmd/1', 'RW_TF/1');
    add_line(actSys, 'RW_TF/1', 'Tau_Out/1');
    add_line(actSys, 'RW_TF/1', 'H_Int/1');
    add_line(actSys, 'H_Int/1', 'Momentum/1');

    %% ===== TOP LEVEL CONNECTIONS =====
    fprintf('  [4/4] Connecting subsystems\n');

    % Command input (step from initial error to zero)
    add_block('simulink/Sources/Constant', [modelName '/Theta_Cmd'], ...
              'Value', '0', 'Position', [50 300 80 320]);

    % Disturbance
    add_block('simulink/Sources/Band-Limited White Noise', [modelName '/Disturbance'], ...
              'Cov', '1e-6', 'Ts', '0.01', 'Position', [50 180 100 210]);

    % Torque sum
    add_block('simulink/Math Operations/Sum', [modelName '/Tau_Sum'], ...
              'Inputs', '+-', 'Position', [120 175 150 205]);

    % Scopes
    add_block('simulink/Sinks/Scope', [modelName '/Theta_Scope'], ...
              'Position', [450 150 480 180]);
    add_block('simulink/Sinks/Scope', [modelName '/Omega_Scope'], ...
              'Position', [450 200 480 230]);
    add_block('simulink/Sinks/Scope', [modelName '/Torque_Scope'], ...
              'Position', [450 250 480 280]);

    % To Workspace
    add_block('simulink/Sinks/To Workspace', [modelName '/Log_Theta'], ...
              'VariableName', 'theta_log', 'Position', [450 300 500 330]);
    add_block('simulink/Sinks/To Workspace', [modelName '/Log_Omega'], ...
              'VariableName', 'omega_log', 'Position', [450 350 500 380]);
    add_block('simulink/Sinks/To Workspace', [modelName '/Log_Torque'], ...
              'VariableName', 'torque_log', 'Position', [450 400 500 430]);

    % Connect the loop
    add_line(modelName, 'Reaction_Wheel/1', 'Tau_Sum/1');
    add_line(modelName, 'Disturbance/1', 'Tau_Sum/2');
    add_line(modelName, 'Tau_Sum/1', 'Attitude_Dynamics/1');
    add_line(modelName, 'Attitude_Dynamics/1', 'PD_Controller/2');
    add_line(modelName, 'Attitude_Dynamics/2', 'PD_Controller/3');
    add_line(modelName, 'Theta_Cmd/1', 'PD_Controller/1');
    add_line(modelName, 'PD_Controller/1', 'Reaction_Wheel/1');

    % Connect to scopes
    add_line(modelName, 'Attitude_Dynamics/1', 'Theta_Scope/1');
    add_line(modelName, 'Attitude_Dynamics/2', 'Omega_Scope/1');
    add_line(modelName, 'Reaction_Wheel/1', 'Torque_Scope/1');

    % Connect to logging
    add_line(modelName, 'Attitude_Dynamics/1', 'Log_Theta/1');
    add_line(modelName, 'Attitude_Dynamics/2', 'Log_Omega/1');
    add_line(modelName, 'Reaction_Wheel/1', 'Log_Torque/1');

    %% ===== SAVE =====
    fprintf('Saving model...\n');
    save_system(modelName);

    fprintf('\n========================================\n');
    fprintf('  Model created: %s.slx\n', modelName);
    fprintf('========================================\n');
    fprintf('To run: sim(''%s'', 100)\n', modelName);
end
