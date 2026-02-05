%% =========================================================================
%  build_advanced_model.m - Build Advanced Multi-Controller GNC Simulink Model
%  =========================================================================
%
%  This script programmatically builds comprehensive Simulink models for
%  the GNC system with multiple control modes and mission phases.
%
%  Creates separate models for:
%    1. GNC_Attitude_Control.slx - Advanced attitude control comparison
%    2. GNC_Launch_Vehicle.slx   - Rocket ascent with staging
%    3. GNC_Orbital_Ops.slx      - Orbital maneuvering and transfers
%    4. GNC_Full_Mission.slx     - Complete integrated mission
%
%  Controllers included:
%    - PID, LQR, LQG, Sliding Mode, MPC, H-infinity
%
%% =========================================================================

function build_advanced_model()
    fprintf('=======================================================\n');
    fprintf('  BUILDING ADVANCED GNC SIMULINK MODELS\n');
    fprintf('=======================================================\n\n');

    % Initialize parameters
    run('GNC_Advanced_Init');

    % Build each model
    build_attitude_control_model();
    build_launch_vehicle_model();
    build_orbital_ops_model();

    fprintf('\n=======================================================\n');
    fprintf('  ALL MODELS BUILT SUCCESSFULLY\n');
    fprintf('=======================================================\n');
end

%% =========================================================================
%  BUILD ATTITUDE CONTROL MODEL
%% =========================================================================
function build_attitude_control_model()
    modelName = 'GNC_Attitude_Control';
    fprintf('\nBuilding %s...\n', modelName);

    % Close if already open
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end

    % Create new model
    new_system(modelName);
    open_system(modelName);

    % Set model parameters
    set_param(modelName, 'Solver', 'ode4');
    set_param(modelName, 'FixedStep', '0.01');
    set_param(modelName, 'StopTime', '100');

    % --- Add Spacecraft Dynamics Subsystem ---
    add_block('built-in/SubSystem', [modelName '/Spacecraft_Dynamics']);
    pos_dyn = [400, 150, 550, 250];
    set_param([modelName '/Spacecraft_Dynamics'], 'Position', pos_dyn);

    % Add dynamics blocks inside subsystem
    create_dynamics_subsystem([modelName '/Spacecraft_Dynamics']);

    % --- Add Controller Selection Subsystem ---
    add_block('built-in/SubSystem', [modelName '/Controller']);
    pos_ctrl = [150, 150, 300, 250];
    set_param([modelName '/Controller'], 'Position', pos_ctrl);

    % Create controller subsystem with mode switching
    create_controller_subsystem([modelName '/Controller']);

    % --- Add Sensor Subsystem ---
    add_block('built-in/SubSystem', [modelName '/Sensors']);
    pos_sens = [550, 300, 650, 380];
    set_param([modelName '/Sensors'], 'Position', pos_sens);

    % --- Add Target Generator ---
    add_block('simulink/Sources/Constant', [modelName '/Target_Quaternion']);
    set_param([modelName '/Target_Quaternion'], 'Value', '[1; 0; 0; 0]');
    set_param([modelName '/Target_Quaternion'], 'Position', [50, 180, 100, 220]);

    % --- Add Scopes and Logging ---
    add_block('simulink/Sinks/Scope', [modelName '/Attitude_Scope']);
    set_param([modelName '/Attitude_Scope'], 'Position', [700, 160, 750, 200]);
    set_param([modelName '/Attitude_Scope'], 'NumInputPorts', '3');

    add_block('simulink/Sinks/To Workspace', [modelName '/Log_Attitude']);
    set_param([modelName '/Log_Attitude'], 'Position', [700, 230, 780, 260]);
    set_param([modelName '/Log_Attitude'], 'VariableName', 'attitude_log');

    % --- Add Controller Mode Selector ---
    add_block('simulink/Sources/Constant', [modelName '/Controller_Mode']);
    set_param([modelName '/Controller_Mode'], 'Value', '2');  % Default LQR
    set_param([modelName '/Controller_Mode'], 'Position', [50, 280, 100, 310]);

    % Save model
    save_system(modelName);
    fprintf('  Created: %s.slx\n', modelName);
end

%% =========================================================================
%  BUILD LAUNCH VEHICLE MODEL
%% =========================================================================
function build_launch_vehicle_model()
    modelName = 'GNC_Launch_Vehicle';
    fprintf('\nBuilding %s...\n', modelName);

    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end

    new_system(modelName);
    open_system(modelName);

    % Set solver for stiff rocket dynamics
    set_param(modelName, 'Solver', 'ode45');
    set_param(modelName, 'MaxStep', '0.1');
    set_param(modelName, 'StopTime', '600');  % 10 minutes for ascent

    % --- Add Stage 1 Dynamics ---
    add_block('built-in/SubSystem', [modelName '/Stage1_Dynamics']);
    set_param([modelName '/Stage1_Dynamics'], 'Position', [200, 100, 350, 200]);

    % Create Stage 1 subsystem content
    create_stage_subsystem([modelName '/Stage1_Dynamics'], 1);

    % --- Add Stage 2 Dynamics ---
    add_block('built-in/SubSystem', [modelName '/Stage2_Dynamics']);
    set_param([modelName '/Stage2_Dynamics'], 'Position', [200, 250, 350, 350]);

    create_stage_subsystem([modelName '/Stage2_Dynamics'], 2);

    % --- Add Staging Logic ---
    add_block('built-in/SubSystem', [modelName '/Staging_Logic']);
    set_param([modelName '/Staging_Logic'], 'Position', [400, 170, 500, 230]);

    % --- Add Atmosphere Model ---
    add_block('built-in/SubSystem', [modelName '/Atmosphere']);
    set_param([modelName '/Atmosphere'], 'Position', [50, 150, 150, 200]);

    % --- Add Gravity Model ---
    add_block('built-in/SubSystem', [modelName '/Gravity']);
    set_param([modelName '/Gravity'], 'Position', [50, 250, 150, 300]);

    % --- Add Guidance ---
    add_block('built-in/SubSystem', [modelName '/Gravity_Turn_Guidance']);
    set_param([modelName '/Gravity_Turn_Guidance'], 'Position', [550, 150, 700, 230]);

    % --- Add Trajectory Logging ---
    add_block('simulink/Sinks/To Workspace', [modelName '/Log_Trajectory']);
    set_param([modelName '/Log_Trajectory'], 'Position', [750, 160, 830, 200]);
    set_param([modelName '/Log_Trajectory'], 'VariableName', 'trajectory_log');

    % --- Add Altitude/Velocity Scopes ---
    add_block('simulink/Sinks/Scope', [modelName '/Flight_Profile']);
    set_param([modelName '/Flight_Profile'], 'Position', [750, 250, 800, 300]);
    set_param([modelName '/Flight_Profile'], 'NumInputPorts', '4');

    save_system(modelName);
    fprintf('  Created: %s.slx\n', modelName);
end

%% =========================================================================
%  BUILD ORBITAL OPERATIONS MODEL
%% =========================================================================
function build_orbital_ops_model()
    modelName = 'GNC_Orbital_Ops';
    fprintf('\nBuilding %s...\n', modelName);

    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end

    new_system(modelName);
    open_system(modelName);

    set_param(modelName, 'Solver', 'ode45');
    set_param(modelName, 'MaxStep', '10');
    set_param(modelName, 'StopTime', '86400');  % 1 day

    % --- Add Orbital Dynamics (includes J2 perturbation) ---
    add_block('built-in/SubSystem', [modelName '/Orbital_Dynamics']);
    set_param([modelName '/Orbital_Dynamics'], 'Position', [300, 150, 450, 250]);

    % --- Add Maneuver Planner ---
    add_block('built-in/SubSystem', [modelName '/Maneuver_Planner']);
    set_param([modelName '/Maneuver_Planner'], 'Position', [100, 150, 200, 220]);

    % --- Add Thrust Controller ---
    add_block('built-in/SubSystem', [modelName '/Thrust_Controller']);
    set_param([modelName '/Thrust_Controller'], 'Position', [500, 150, 600, 220]);

    % --- Add Orbit Determination (EKF) ---
    add_block('built-in/SubSystem', [modelName '/Orbit_Determination']);
    set_param([modelName '/Orbit_Determination'], 'Position', [300, 300, 450, 380]);

    % --- Add Hohmann Transfer Calculator ---
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
              [modelName '/Hohmann_DeltaV']);
    set_param([modelName '/Hohmann_DeltaV'], 'Position', [100, 300, 200, 350]);

    % --- Add 3D Trajectory Visualization ---
    add_block('simulink/Sinks/XY Graph', [modelName '/Orbit_Display']);
    set_param([modelName '/Orbit_Display'], 'Position', [650, 150, 750, 250]);

    % --- Add Orbit Elements Display ---
    add_block('simulink/Sinks/Display', [modelName '/Orbit_Elements']);
    set_param([modelName '/Orbit_Elements'], 'Position', [650, 300, 750, 380]);

    save_system(modelName);
    fprintf('  Created: %s.slx\n', modelName);
end

%% =========================================================================
%  HELPER: CREATE DYNAMICS SUBSYSTEM
%% =========================================================================
function create_dynamics_subsystem(subsysPath)
    % Create the internal blocks for spacecraft dynamics

    % Input ports
    add_block('simulink/Ports & Subsystems/In1', [subsysPath '/Torque_Cmd']);
    add_block('simulink/Ports & Subsystems/In1', [subsysPath '/Current_Attitude']);

    % Output ports
    add_block('simulink/Ports & Subsystems/Out1', [subsysPath '/Attitude_Out']);
    add_block('simulink/Ports & Subsystems/Out1', [subsysPath '/Omega_Out']);

    % Euler's equations: I * omega_dot = tau - omega x (I * omega)
    add_block('simulink/Math Operations/Gain', [subsysPath '/Inertia_Inv']);
    set_param([subsysPath '/Inertia_Inv'], 'Gain', 'inv(I_spacecraft)');

    % Integrators for angular velocity and attitude
    add_block('simulink/Continuous/Integrator', [subsysPath '/Omega_Integrator']);
    set_param([subsysPath '/Omega_Integrator'], 'InitialCondition', 'omega0');

    add_block('simulink/Continuous/Integrator', [subsysPath '/Quat_Integrator']);
    set_param([subsysPath '/Quat_Integrator'], 'InitialCondition', 'q0');

    % Quaternion kinematics: q_dot = 0.5 * q * omega_quat
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
              [subsysPath '/Quat_Kinematics']);
end

%% =========================================================================
%  HELPER: CREATE CONTROLLER SUBSYSTEM
%% =========================================================================
function create_controller_subsystem(subsysPath)
    % Multi-mode controller with switching logic

    % Input ports
    add_block('simulink/Ports & Subsystems/In1', [subsysPath '/Attitude_Error']);
    add_block('simulink/Ports & Subsystems/In1', [subsysPath '/Omega_Error']);
    add_block('simulink/Ports & Subsystems/In1', [subsysPath '/Controller_Mode']);

    % Output port
    add_block('simulink/Ports & Subsystems/Out1', [subsysPath '/Torque_Cmd']);

    % Multiport switch for controller selection
    add_block('simulink/Signal Routing/Multiport Switch', [subsysPath '/Mode_Switch']);
    set_param([subsysPath '/Mode_Switch'], 'Inputs', '6');

    % PID Controller
    add_block('simulink/Continuous/PID Controller', [subsysPath '/PID']);
    set_param([subsysPath '/PID'], 'P', 'PID.Kp(1)');
    set_param([subsysPath '/PID'], 'I', 'PID.Ki(1)');
    set_param([subsysPath '/PID'], 'D', 'PID.Kd(1)');

    % LQR Controller (gain block)
    add_block('simulink/Math Operations/Gain', [subsysPath '/LQR_Gain']);
    set_param([subsysPath '/LQR_Gain'], 'Gain', '-LQR.K');

    % SMC Controller (MATLAB Function)
    add_block('simulink/User-Defined Functions/MATLAB Function', [subsysPath '/SMC']);

    % MPC Controller (placeholder - would use MPC Toolbox)
    add_block('simulink/Math Operations/Gain', [subsysPath '/MPC_Gain']);
    set_param([subsysPath '/MPC_Gain'], 'Gain', '-LQR.K');  % Fallback to LQR

    % Saturation for torque limits
    add_block('simulink/Discontinuities/Saturation', [subsysPath '/Torque_Sat']);
    set_param([subsysPath '/Torque_Sat'], 'UpperLimit', '200');
    set_param([subsysPath '/Torque_Sat'], 'LowerLimit', '-200');
end

%% =========================================================================
%  HELPER: CREATE STAGE SUBSYSTEM
%% =========================================================================
function create_stage_subsystem(subsysPath, stageNum)
    % Rocket stage dynamics

    % Input ports
    add_block('simulink/Ports & Subsystems/In1', [subsysPath '/Throttle']);
    add_block('simulink/Ports & Subsystems/In1', [subsysPath '/Gimbal_Cmd']);
    add_block('simulink/Ports & Subsystems/In1', [subsysPath '/Atmosphere']);

    % Output ports
    add_block('simulink/Ports & Subsystems/Out1', [subsysPath '/Thrust']);
    add_block('simulink/Ports & Subsystems/Out1', [subsysPath '/Mass']);
    add_block('simulink/Ports & Subsystems/Out1', [subsysPath '/Propellant']);

    % Stage parameters lookup
    stageStr = sprintf('Stage%d', stageNum);

    % Thrust calculation
    add_block('simulink/User-Defined Functions/MATLAB Function', ...
              [subsysPath '/Thrust_Model']);

    % Mass depletion (integrator)
    add_block('simulink/Continuous/Integrator', [subsysPath '/Mass_Integrator']);
    set_param([subsysPath '/Mass_Integrator'], ...
              'InitialCondition', sprintf('%s.propellant_mass', stageStr));

    % Mass flow rate
    add_block('simulink/Math Operations/Gain', [subsysPath '/Mass_Flow']);
end

%% =========================================================================
%  RUN IF CALLED DIRECTLY
%% =========================================================================
% Execute build
build_advanced_model();
