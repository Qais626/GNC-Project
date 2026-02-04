// =============================================================================
// hil_interface.cpp -- Hardware-in-the-Loop Interface Implementation
// =============================================================================
//
// This file implements the HILInterface class for simulating hardware-in-the-
// loop communication between a GNC simulation and flight hardware.
//
// IMPLEMENTATION STRATEGY: MOCK MODE
//
//   Since we don't have actual flight hardware connected, this implementation
//   operates in "mock mode" -- it generates synthetic sensor data that
//   mimics what real hardware would produce during HIL testing.
//
//   The mock data simulates a spacecraft in a circular LEO orbit:
//     - IMU (gyro + accelerometer) with realistic noise and bias
//     - GPS position/velocity with dilution of precision
//     - Star tracker quaternions with measurement uncertainty
//     - Sun sensor direction in body frame
//
//   The mock mode is invaluable for:
//     - Developing and debugging the HIL interface before hardware arrives
//     - Regression testing without hardware in the loop
//     - Performance benchmarking of the communication layer
//     - Training operators on HIL procedures
//
// CRC-16/CCITT IMPLEMENTATION:
//
//   We use the CRC-16/CCITT polynomial (0x1021) with initial value 0xFFFF.
//   This is the standard checksum for aerospace telemetry:
//     - CCSDS (Consultative Committee for Space Data Systems)
//     - MIL-STD-1553 (military avionics bus)
//     - SpaceWire (ESA high-speed spacecraft interconnect)
//
//   The lookup table approach computes CRC one byte at a time using a
//   pre-computed 256-entry table. This is ~8x faster than bit-by-bit
//   computation, and the 512-byte table fits entirely in L1 cache.
//
// REAL-TIME LOOP:
//
//   The run_hil_loop() method implements a precise timing loop using
//   std::chrono::steady_clock. It maintains the commanded rate (e.g., 50 Hz)
//   by computing the next deadline and sleeping until that time.
//
//   This approach is superior to fixed sleep_for() because it doesn't
//   accumulate drift: if one cycle runs long, the next sleep is shorter
//   to compensate. Over N cycles, the total elapsed time is exactly N*dt.
//
// =============================================================================

#include "hil_interface.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

// ---------------------------------------------------------------------------
// CRC-16/CCITT Lookup Table
//
// Polynomial: 0x1021 (x^16 + x^12 + x^5 + 1)
// Initial value: 0xFFFF
//
// This table is pre-computed at compile time. Each entry T[i] represents
// the CRC contribution of byte value i. To compute the CRC of a message:
//   crc = 0xFFFF
//   for each byte b in message:
//     crc = (crc << 8) ^ T[(crc >> 8) ^ b]
//
// The table is 256 entries * 2 bytes = 512 bytes, fitting in a single
// cache line group. On the first CRC computation, the entire table is
// loaded into L1 cache and stays there for subsequent computations.
//
// GENERATION: Each entry is computed by processing byte value i through
// 8 rounds of the CRC division algorithm:
//   for bit 7 down to 0:
//     if MSB of crc is 1: crc = (crc << 1) ^ 0x1021
//     else: crc = crc << 1
// ---------------------------------------------------------------------------
static const uint16_t crc16_ccitt_table[256] = {
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
    0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
    0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485,
    0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4,
    0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
    0x4864, 0x5845, 0x6826, 0x7807, 0x08E0, 0x18C1, 0x28A2, 0x38A3,
    0xC94C, 0xD96D, 0xE90E, 0xF92F, 0x89C8, 0x99E9, 0xA98A, 0xB9AB,
    0x5A75, 0x4A54, 0x7A37, 0x6A16, 0x1AF1, 0x0AD0, 0x3AB3, 0x2A92,
    0xDB7D, 0xCB5C, 0xFB3F, 0xEB1E, 0x9BF9, 0x8BD8, 0xBBBB, 0xAB9A,
    0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41,
    0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
    0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
    0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78,
    0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F,
    0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E,
    0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
    0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E, 0xC71D, 0xD73C,
    0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3,
    0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A,
    0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
    0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9,
    0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
    0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8,
    0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0
};

// ---------------------------------------------------------------------------
// Constructor
//
// Initializes the HIL interface in mock mode (no real hardware).
// All communication will be simulated locally, generating synthetic
// sensor data based on a truth orbit model.
//
// The mock hardware is initialized with:
//   - 500 us simulated communication delay (realistic for serial at 115200 baud)
//   - 0.001 standard deviation for sensor noise (realistic for MEMS IMU)
//   - Sequence counter starting at 0
// ---------------------------------------------------------------------------
HILInterface::HILInterface()
    : socket_fd_(-1),
      connected_(false),
      tx_sequence_(0),
      clock_offset_us_(0.0),
      one_way_delay_us_(0.0)
{
    // Initialize mock hardware settings
    mock_.enabled = true;   // Default to mock mode (no real hardware)
    mock_.sim_delay_us = 500.0;
    mock_.noise_std = 0.001;
    mock_.sequence = 0;

    // Zero-initialize the last sensor packet
    std::memset(&mock_.last_sensor, 0, sizeof(SensorPacket));

    // Initialize statistics to zero
    stats_.packets_sent = 0;
    stats_.packets_received = 0;
    stats_.packets_dropped = 0;
    stats_.crc_errors = 0;
    stats_.timeout_count = 0;
    stats_.clock_offset_us = 0.0;
    stats_.one_way_delay_us = 0.0;
    stats_.avg_round_trip_us = 0.0;
    stats_.link_uptime_s = 0.0;
    stats_.connected = false;

    // Pre-allocate log vector to avoid heap allocations during operation
    log_.reserve(10000);

    std::printf("[HIL] HILInterface created (mock mode: %s)\n",
                mock_.enabled ? "ENABLED" : "disabled");
}

// ---------------------------------------------------------------------------
// Destructor
//
// Ensures clean disconnection. Closes socket if open, flushes logs.
// ---------------------------------------------------------------------------
HILInterface::~HILInterface() {
    if (connected_) {
        disconnect();
    }
}

// ---------------------------------------------------------------------------
// connect() (with HILConfig)
//
// Establishes connection to the HIL hardware (or mock).
//
// In mock mode:
//   - No actual network or serial connection is opened
//   - A simulated connection is established instantly
//   - Mock sensor data generation begins
//
// In real mode (future implementation):
//   - For UDP: creates a socket, binds to rx port, connects to tx
//   - For serial: opens the device, configures baud rate, parity, etc.
// ---------------------------------------------------------------------------
bool HILInterface::connect(const HILConfig& config) {
    config_ = config;

    if (mock_.enabled) {
        std::printf("[HIL] Mock HIL connected (simulating %s on %s)\n",
                    config.protocol == HILConfig::Protocol::UDP ? "UDP" : "Serial",
                    config.protocol == HILConfig::Protocol::UDP ?
                        config.udp_host.c_str() : config.serial_port.c_str());

        connected_ = true;
        stats_.connected = true;
        connect_time_ = std::chrono::steady_clock::now();

        // Generate initial mock sensor data
        std::printf("[HIL] Generating mock sensor data for circular LEO orbit\n");

        return true;
    }

    // Real hardware connection would go here.
    // For UDP:
    //   socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    //   bind() to config.udp_port_rx
    //   Set receive timeout using setsockopt(SO_RCVTIMEO)
    //
    // For Serial:
    //   socket_fd_ = open(config.serial_port.c_str(), O_RDWR | O_NOCTTY);
    //   Configure with termios: baud rate, 8N1, raw mode
    //   tcflush() to clear any stale data

    std::fprintf(stderr, "[HIL] Real hardware connection not implemented\n");
    return false;
}

// ---------------------------------------------------------------------------
// connect() (convenience overload with simple parameters)
// ---------------------------------------------------------------------------
bool HILInterface::connect(const std::string& port, uint32_t baud_rate) {
    HILConfig config;
    config.protocol = HILConfig::Protocol::SERIAL;
    config.serial_port = port;
    config.baud_rate = baud_rate;
    return connect(config);
}

// ---------------------------------------------------------------------------
// disconnect()
//
// Closes the HIL connection and logs final statistics.
// ---------------------------------------------------------------------------
void HILInterface::disconnect() {
    if (!connected_) return;

    auto now = std::chrono::steady_clock::now();
    double uptime = std::chrono::duration<double>(now - connect_time_).count();

    std::printf("[HIL] Disconnecting (uptime: %.2f s, sent: %zu, received: %zu, "
                "dropped: %zu, CRC errors: %zu)\n",
                uptime, stats_.packets_sent, stats_.packets_received,
                stats_.packets_dropped, stats_.crc_errors);

    connected_ = false;
    stats_.connected = false;
    stats_.link_uptime_s = uptime;

    // Close socket if open (real mode)
    if (socket_fd_ >= 0) {
        // close(socket_fd_);
        socket_fd_ = -1;
    }
}

// ---------------------------------------------------------------------------
// is_connected()
// ---------------------------------------------------------------------------
bool HILInterface::is_connected() const {
    return connected_;
}

// ---------------------------------------------------------------------------
// MockHardware::generate_sensor_data()
//
// Generates a synthetic SensorPacket that simulates what real hardware
// would produce during a circular LEO orbit.
//
// The simulated orbit:
//   - Altitude: 400 km (ISS-like)
//   - Inclination: 51.6 degrees (ISS)
//   - Period: ~92 minutes
//
// Sensor models:
//   - Gyroscope: true angular rate + bias + white noise
//     Bias: 0.01 rad/s (typical MEMS gyro bias, 0.5 deg/s)
//     Noise: 0.001 rad/s RMS (typical MEMS ARW)
//
//   - Accelerometer: specific force + bias + white noise
//     In orbit, the accelerometer reads ~0 (freefall) plus drag
//     Noise: 0.01 m/s^2 RMS
//
//   - GPS: position + white noise (3-10 m RMS depending on PDOP)
//
//   - Star tracker: quaternion with small attitude error (~10 arcsec)
//
//   - Sun sensor: sun vector in body frame (valid when sun is visible)
//
// NOISE GENERATION:
//   We use a simple linear congruential generator (LCG) for deterministic
//   pseudo-random noise. This is NOT cryptographically secure, but it's
//   fast (single multiply-add per sample) and produces noise with
//   sufficient statistical quality for sensor simulation.
// ---------------------------------------------------------------------------
SensorPacket HILInterface::MockHardware::generate_sensor_data(double time_s) {
    SensorPacket packet;
    std::memset(&packet, 0, sizeof(SensorPacket));

    // Packet header
    packet.header.sync = SYNC_BYTE;
    packet.header.type = PacketType::SENSOR_DATA;
    packet.header.sequence = sequence++;
    packet.header.payload_len = sizeof(SensorPacket) - sizeof(PacketHeader) - sizeof(uint16_t);
    packet.header.reserved = 0;

    // Hardware timestamp (microseconds since boot)
    packet.hw_timestamp_us = static_cast<uint64_t>(time_s * 1.0e6);

    // Orbital parameters for a circular LEO orbit
    constexpr double R_orbit = 6.778137e6;  // 400 km altitude (m)
    constexpr double mu = 3.986004418e14;   // Earth mu (m^3/s^2)
    const double orbital_rate = std::sqrt(mu / (R_orbit * R_orbit * R_orbit));  // rad/s
    const double inclination = 51.6 * 3.14159265358979 / 180.0;  // ISS inclination

    // Compute true position in ECI frame
    double angle = orbital_rate * time_s;
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);
    double cos_i = std::cos(inclination);
    double sin_i = std::sin(inclination);

    // Simple pseudo-random noise generator (deterministic for repeatability).
    // Uses a hash of the sequence number to produce different noise each packet.
    // The noise is approximately Gaussian via the sum of uniform variates
    // (central limit theorem with 4 terms).
    auto noise = [&](double std_dev) -> double {
        // Simple deterministic noise based on time and a seed
        uint32_t seed = static_cast<uint32_t>(time_s * 1000.0) * 2654435761u;
        seed ^= (seed >> 16);
        seed *= 0x45d9f3b;
        seed ^= (seed >> 16);
        // Map to [-1, 1] and scale by standard deviation
        double u1 = (static_cast<double>(seed & 0xFFFF) / 65536.0 - 0.5) * 2.0;
        seed *= 1103515245u;
        seed += 12345u;
        double u2 = (static_cast<double>(seed & 0xFFFF) / 65536.0 - 0.5) * 2.0;
        // Approximate Gaussian: sum of two uniform / sqrt(2) ~ (0, 0.816)
        return (u1 + u2) * 0.5 * std_dev;
    };

    // --- IMU Data ---
    // Gyroscope: body angular rates during nadir-pointing attitude
    // For nadir pointing, the body rotates at orbital rate around one axis
    packet.gyro[0] = static_cast<float>(0.0 + noise(noise_std));            // wx (rad/s)
    packet.gyro[1] = static_cast<float>(-orbital_rate + noise(noise_std));   // wy (orbital rate)
    packet.gyro[2] = static_cast<float>(0.0 + noise(noise_std));            // wz

    // Accelerometer: in freefall, reads ~0 (plus drag at LEO altitudes)
    // Atmospheric drag at 400 km: ~1e-6 m/s^2 (very small)
    packet.accel[0] = static_cast<float>(0.0 + noise(noise_std * 10.0));
    packet.accel[1] = static_cast<float>(0.0 + noise(noise_std * 10.0));
    packet.accel[2] = static_cast<float>(0.0 + noise(noise_std * 10.0));

    // Gyro temperature (affects bias stability)
    packet.gyro_temp = static_cast<float>(25.0 + 5.0 * std::sin(angle * 0.1));

    // --- GPS Data ---
    // ECEF position from orbital elements
    double x = R_orbit * cos_a;
    double y = R_orbit * sin_a * cos_i;
    double z = R_orbit * sin_a * sin_i;

    packet.gps_pos_ecef[0] = x + noise(5.0);   // 5m RMS position noise
    packet.gps_pos_ecef[1] = y + noise(5.0);
    packet.gps_pos_ecef[2] = z + noise(5.0);

    // ECEF velocity (derivative of position)
    double v = std::sqrt(mu / R_orbit);  // Circular velocity
    double vx = -v * sin_a;
    double vy =  v * cos_a * cos_i;
    double vz =  v * cos_a * sin_i;

    packet.gps_vel_ecef[0] = vx + noise(0.1);  // 0.1 m/s RMS velocity noise
    packet.gps_vel_ecef[1] = vy + noise(0.1);
    packet.gps_vel_ecef[2] = vz + noise(0.1);

    // GPS quality indicators
    packet.gps_pdop = static_cast<float>(1.5 + std::fabs(noise(0.5)));
    packet.gps_num_sats = 8 + static_cast<uint8_t>(std::fabs(noise(3.0)));
    packet.gps_fix_type = 2;  // 3D fix

    // --- Star Tracker ---
    // Nadir-pointing quaternion with small measurement error
    // For a nadir-pointing orbit, the body z-axis points toward Earth center
    double half_angle = angle * 0.5;
    packet.star_quat[0] = static_cast<float>(std::sin(half_angle) * 0.0 + noise(1e-4));
    packet.star_quat[1] = static_cast<float>(std::sin(half_angle) * 1.0 + noise(1e-4));
    packet.star_quat[2] = static_cast<float>(0.0 + noise(1e-4));
    packet.star_quat[3] = static_cast<float>(std::cos(half_angle));

    // Normalize the quaternion
    float qnorm = std::sqrt(packet.star_quat[0] * packet.star_quat[0] +
                             packet.star_quat[1] * packet.star_quat[1] +
                             packet.star_quat[2] * packet.star_quat[2] +
                             packet.star_quat[3] * packet.star_quat[3]);
    if (qnorm > 1e-6f) {
        for (int i = 0; i < 4; ++i) packet.star_quat[i] /= qnorm;
    }

    packet.star_confidence = static_cast<float>(0.95 + noise(0.02));
    packet.star_num_stars = 12 + static_cast<uint8_t>(std::fabs(noise(4.0)));

    // --- Sun Sensor ---
    // Sun direction in body frame (simplified: sun always at +X in ECI)
    // Sun visibility depends on whether spacecraft is in eclipse
    double sun_angle = angle;  // Simplified sun-spacecraft geometry
    bool in_eclipse = (std::sin(sun_angle) < -0.3);

    if (!in_eclipse) {
        packet.sun_vector[0] = static_cast<float>(std::cos(sun_angle) + noise(0.01));
        packet.sun_vector[1] = static_cast<float>(std::sin(sun_angle) * cos_i + noise(0.01));
        packet.sun_vector[2] = static_cast<float>(std::sin(sun_angle) * sin_i + noise(0.01));

        // Normalize sun vector
        float sun_norm = std::sqrt(packet.sun_vector[0] * packet.sun_vector[0] +
                                   packet.sun_vector[1] * packet.sun_vector[1] +
                                   packet.sun_vector[2] * packet.sun_vector[2]);
        if (sun_norm > 1e-6f) {
            for (int i = 0; i < 3; ++i) packet.sun_vector[i] /= sun_norm;
        }
        packet.sun_valid = 1;
    } else {
        packet.sun_vector[0] = 0.0f;
        packet.sun_vector[1] = 0.0f;
        packet.sun_vector[2] = 0.0f;
        packet.sun_valid = 0;
    }

    // Compute CRC over entire packet (excluding the CRC field itself)
    std::size_t crc_len = sizeof(SensorPacket) - sizeof(uint16_t);
    packet.crc = HILInterface::compute_crc16(
        reinterpret_cast<const uint8_t*>(&packet), crc_len);

    last_sensor = packet;
    return packet;
}

// ---------------------------------------------------------------------------
// send_actuator_commands()
//
// Sends an actuator command packet to the flight hardware.
//
// In mock mode: the command is logged but not transmitted.
// In real mode: the packet is serialized, CRC is appended, and it's sent
// via the configured transport (UDP or serial).
//
// PACKET FORMAT:
//   [sync byte] [packet type] [sequence] [payload length] [payload] [CRC-16]
//
// The actuator command includes:
//   - 4 reaction wheel torque/speed commands
//   - 8 thruster on/off commands
//   - 3 magnetic torquer dipole commands
// ---------------------------------------------------------------------------
bool HILInterface::send_actuator_commands(const ActuatorCommandPacket& cmds) {
    if (!connected_) {
        std::fprintf(stderr, "[HIL] Cannot send: not connected\n");
        return false;
    }

    // Build the packet with header and CRC
    ActuatorCommandPacket packet = cmds;
    packet.header.sync = SYNC_BYTE;
    packet.header.type = PacketType::ACTUATOR_CMD;
    packet.header.sequence = tx_sequence_++;
    packet.header.payload_len = sizeof(ActuatorCommandPacket) - sizeof(PacketHeader) - sizeof(uint16_t);

    // Compute CRC
    std::size_t crc_len = sizeof(ActuatorCommandPacket) - sizeof(uint16_t);
    packet.crc = compute_crc16(reinterpret_cast<const uint8_t*>(&packet), crc_len);

    // Log the event
    log_event(HILLogEntry::Direction::SEND, PacketType::ACTUATOR_CMD,
              packet.header.sequence, sizeof(ActuatorCommandPacket), true);
    stats_.packets_sent++;

    if (mock_.enabled) {
        // In mock mode, just log the command (no actual transmission)
        return true;
    }

    // Real mode: transmit the packet
    return send_raw(reinterpret_cast<const uint8_t*>(&packet), sizeof(ActuatorCommandPacket));
}

// ---------------------------------------------------------------------------
// receive_sensor_data()
//
// Receives a sensor data packet from the flight hardware.
//
// In mock mode: generates synthetic sensor data using the orbit model.
// In real mode: reads from the socket/serial port with timeout.
//
// Returns std::optional<SensorPacket>: the packet if received successfully,
// or std::nullopt if the receive timed out or CRC validation failed.
// ---------------------------------------------------------------------------
std::optional<SensorPacket> HILInterface::receive_sensor_data() {
    if (!connected_) {
        return std::nullopt;
    }

    if (mock_.enabled) {
        // Generate mock sensor data at the current simulation time
        auto now = std::chrono::steady_clock::now();
        double time_s = std::chrono::duration<double>(now - connect_time_).count();

        SensorPacket packet = mock_.generate_sensor_data(time_s);

        // Verify CRC (should always pass for mock data, but validates our CRC code)
        std::size_t crc_len = sizeof(SensorPacket) - sizeof(uint16_t);
        uint16_t computed_crc = compute_crc16(
            reinterpret_cast<const uint8_t*>(&packet), crc_len);

        bool crc_valid = (computed_crc == packet.crc);
        if (!crc_valid) {
            stats_.crc_errors++;
            std::fprintf(stderr, "[HIL] CRC error in mock data (computed: 0x%04X, "
                         "received: 0x%04X)\n", computed_crc, packet.crc);
        }

        // Log the event
        log_event(HILLogEntry::Direction::RECEIVE, PacketType::SENSOR_DATA,
                  packet.header.sequence, sizeof(SensorPacket), crc_valid);
        stats_.packets_received++;

        // Call receive callback if registered
        if (receive_callback_) {
            receive_callback_(packet);
        }

        return packet;
    }

    // Real mode: read from transport (not yet implemented)
    // uint8_t buffer[512];
    // int bytes = receive_raw(buffer, sizeof(buffer), config_.timeout_ms);
    // if (bytes < 0) { stats_.timeout_count++; return std::nullopt; }
    // ... parse packet, validate CRC, return ...

    return std::nullopt;
}

// ---------------------------------------------------------------------------
// send_health_request()
// ---------------------------------------------------------------------------
bool HILInterface::send_health_request() {
    if (!connected_) return false;

    // In mock mode, this is a no-op (health status is always available)
    stats_.packets_sent++;
    return true;
}

// ---------------------------------------------------------------------------
// receive_health_status()
// ---------------------------------------------------------------------------
std::optional<HealthPacket> HILInterface::receive_health_status() {
    if (!connected_) return std::nullopt;

    if (mock_.enabled) {
        HealthPacket packet;
        std::memset(&packet, 0, sizeof(HealthPacket));

        packet.header.sync = SYNC_BYTE;
        packet.header.type = PacketType::HEALTH_STATUS;
        packet.header.sequence = tx_sequence_++;
        packet.header.payload_len = sizeof(HealthPacket) - sizeof(PacketHeader) - sizeof(uint16_t);

        packet.mode = 3;  // NOMINAL
        packet.cpu_temp = 42.5f;
        packet.bus_voltage = 28.0f;
        packet.bus_current = 2.5f;

        auto now = std::chrono::steady_clock::now();
        packet.uptime_s = static_cast<uint32_t>(
            std::chrono::duration<double>(now - connect_time_).count());
        packet.error_count = 0;

        std::size_t crc_len = sizeof(HealthPacket) - sizeof(uint16_t);
        packet.crc = compute_crc16(reinterpret_cast<const uint8_t*>(&packet), crc_len);

        stats_.packets_received++;
        return packet;
    }

    return std::nullopt;
}

// ---------------------------------------------------------------------------
// perform_time_sync()
//
// Performs a timestamp synchronization exchange using a simplified PTP
// (Precision Time Protocol) scheme.
//
// In mock mode, the clock offset is simulated as zero (both sides run
// on the same CPU clock). A small random jitter is added to simulate
// real-world conditions.
// ---------------------------------------------------------------------------
bool HILInterface::perform_time_sync() {
    if (!connected_) return false;

    if (mock_.enabled) {
        // Simulate time sync: in mock mode, offset is effectively zero
        // Add small simulated jitter
        clock_offset_us_ = 0.5;   // 0.5 us simulated offset
        one_way_delay_us_ = mock_.sim_delay_us;  // Simulated one-way delay

        stats_.clock_offset_us = clock_offset_us_;
        stats_.one_way_delay_us = one_way_delay_us_;
        stats_.avg_round_trip_us = 2.0 * one_way_delay_us_;

        return true;
    }

    // Real PTP implementation:
    // 1. Record T1 = now()
    // 2. Send TimeSyncPacket with T1
    // 3. Receive response with T2, T3
    // 4. Record T4 = now()
    // 5. offset = ((T2-T1) + (T3-T4)) / 2
    // 6. delay  = ((T4-T1) - (T3-T2)) / 2

    return false;
}

// ---------------------------------------------------------------------------
// get_clock_offset_us()
// ---------------------------------------------------------------------------
double HILInterface::get_clock_offset_us() const {
    return clock_offset_us_;
}

// ---------------------------------------------------------------------------
// hw_to_sim_time()
//
// Converts a hardware timestamp to simulation time by applying the
// estimated clock offset from the most recent time sync.
// ---------------------------------------------------------------------------
double HILInterface::hw_to_sim_time(uint64_t hw_timestamp_us) const {
    return static_cast<double>(hw_timestamp_us) - clock_offset_us_;
}

// ---------------------------------------------------------------------------
// compute_crc16()
//
// CRC-16/CCITT computation using the lookup table method.
//
// Algorithm:
//   1. Initialize CRC to 0xFFFF
//   2. For each byte in the data:
//      a. Compute table index = (CRC >> 8) XOR byte
//      b. CRC = (CRC << 8) XOR table[index]
//   3. Return final CRC value
//
// PERFORMANCE: This processes one byte per iteration with ~5 operations
// (shift, XOR, table lookup, shift, XOR). At 1 GHz, that's ~5 ns/byte
// or ~200 MB/s throughput. For our small packets (~100 bytes), the CRC
// computation takes ~0.5 us -- negligible compared to communication latency.
//
// ALTERNATIVE: Some CPUs have hardware CRC instructions (CRC32C on x86
// via SSE4.2). These process 8 bytes per cycle for ~10 GB/s throughput.
// However, they use the CRC-32C polynomial, not CRC-16/CCITT.
// ---------------------------------------------------------------------------
uint16_t HILInterface::compute_crc16(const uint8_t* data, std::size_t len) {
    uint16_t crc = 0xFFFF;  // Initial value per CCITT specification

    for (std::size_t i = 0; i < len; ++i) {
        // The table index combines the high byte of the current CRC
        // with the input data byte. This effectively processes 8 bits
        // of data per iteration instead of 1 bit at a time.
        uint8_t index = static_cast<uint8_t>((crc >> 8) ^ data[i]);
        crc = static_cast<uint16_t>((crc << 8) ^ crc16_ccitt_table[index]);
    }

    return crc;
}

// ---------------------------------------------------------------------------
// validate_packet_crc()
//
// Validates a packet's CRC by recomputing it over the data portion
// and comparing with the CRC field at the end.
//
// The CRC field is the last 2 bytes of the packet. We compute CRC over
// everything except those last 2 bytes and compare.
// ---------------------------------------------------------------------------
bool HILInterface::validate_packet_crc(const uint8_t* packet, std::size_t len) {
    if (len < 4) return false;  // Too short to contain header + CRC

    // CRC is the last 2 bytes of the packet
    uint16_t received_crc = static_cast<uint16_t>(
        (static_cast<uint16_t>(packet[len - 1]) << 8) | packet[len - 2]);

    // Compute CRC over everything except the CRC field
    uint16_t computed_crc = compute_crc16(packet, len - 2);

    return computed_crc == received_crc;
}

// ---------------------------------------------------------------------------
// Low-level I/O (stubs for mock mode)
// ---------------------------------------------------------------------------
bool HILInterface::send_raw(const uint8_t* /*data*/, std::size_t /*len*/) {
    // In a real implementation:
    //   For UDP: sendto(socket_fd_, data, len, 0, &addr, sizeof(addr))
    //   For serial: write(socket_fd_, data, len)
    return mock_.enabled;
}

int HILInterface::receive_raw(uint8_t* /*buffer*/, std::size_t /*max_len*/, double /*timeout_ms*/) {
    // In a real implementation:
    //   For UDP: recvfrom() with select() for timeout
    //   For serial: read() with select() for timeout
    return mock_.enabled ? 0 : -1;
}

// ---------------------------------------------------------------------------
// encode_packet() / decode_packet()
//
// Packet serialization/deserialization for the wire protocol.
// ---------------------------------------------------------------------------
std::vector<uint8_t> HILInterface::encode_packet(
    PacketType type,
    const uint8_t* payload,
    std::size_t payload_len)
{
    std::size_t total_len = sizeof(PacketHeader) + payload_len + sizeof(uint16_t);
    std::vector<uint8_t> buffer(total_len);

    // Build header
    PacketHeader header;
    header.sync = SYNC_BYTE;
    header.type = type;
    header.sequence = tx_sequence_++;
    header.payload_len = static_cast<uint16_t>(payload_len);
    header.reserved = 0;

    // Copy header
    std::memcpy(buffer.data(), &header, sizeof(PacketHeader));

    // Copy payload
    if (payload && payload_len > 0) {
        std::memcpy(buffer.data() + sizeof(PacketHeader), payload, payload_len);
    }

    // Compute and append CRC
    uint16_t crc = compute_crc16(buffer.data(), total_len - sizeof(uint16_t));
    std::memcpy(buffer.data() + total_len - sizeof(uint16_t), &crc, sizeof(uint16_t));

    return buffer;
}

bool HILInterface::decode_packet(
    const uint8_t* data,
    std::size_t len,
    PacketHeader& header,
    const uint8_t*& payload)
{
    if (len < sizeof(PacketHeader) + sizeof(uint16_t)) {
        return false;  // Packet too short
    }

    // Extract header
    std::memcpy(&header, data, sizeof(PacketHeader));

    // Validate sync byte
    if (header.sync != SYNC_BYTE) {
        return false;
    }

    // Validate length
    if (sizeof(PacketHeader) + header.payload_len + sizeof(uint16_t) > len) {
        return false;
    }

    // Validate CRC
    std::size_t crc_data_len = sizeof(PacketHeader) + header.payload_len;
    uint16_t received_crc;
    std::memcpy(&received_crc, data + crc_data_len, sizeof(uint16_t));
    uint16_t computed_crc = compute_crc16(data, crc_data_len);

    if (received_crc != computed_crc) {
        stats_.crc_errors++;
        return false;
    }

    payload = data + sizeof(PacketHeader);
    return true;
}

// ---------------------------------------------------------------------------
// log_event()
//
// Records a packet event for post-test analysis.
// The log is stored in memory (pre-allocated vector) for minimal overhead.
// In a real system, this would also feed the telemetry downlink.
// ---------------------------------------------------------------------------
void HILInterface::log_event(
    HILLogEntry::Direction dir,
    PacketType type,
    uint16_t seq,
    std::size_t size,
    bool crc_ok)
{
    auto now = std::chrono::steady_clock::now();
    double timestamp = std::chrono::duration<double>(now - connect_time_).count();

    HILLogEntry entry;
    entry.timestamp = timestamp;
    entry.direction = dir;
    entry.type = type;
    entry.sequence = seq;
    entry.size_bytes = size;
    entry.crc_valid = crc_ok;

    log_.push_back(entry);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
HILStats HILInterface::get_stats() const {
    HILStats s = stats_;
    if (connected_) {
        auto now = std::chrono::steady_clock::now();
        s.link_uptime_s = std::chrono::duration<double>(now - connect_time_).count();
    }
    return s;
}

std::vector<HILLogEntry> HILInterface::get_log() const {
    return log_;
}

void HILInterface::clear_log() {
    log_.clear();
}

void HILInterface::set_receive_callback(std::function<void(const SensorPacket&)> cb) {
    receive_callback_ = std::move(cb);
}
