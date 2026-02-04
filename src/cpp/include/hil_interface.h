// =============================================================================
// hil_interface.h -- Hardware-in-the-Loop Interface
// =============================================================================
//
// OVERVIEW:
//   Provides the communication layer between the GNC simulation and real
//   hardware (sensors, actuators, flight computers) for hardware-in-the-loop
//   (HIL) testing.
//
// HIL TESTING IN AEROSPACE:
//
//   HIL testing connects real flight hardware to a simulated environment:
//
//   +------------------+       +------------------+
//   |  Simulation PC   |<----->|  Flight Hardware  |
//   |                  |       |                  |
//   | - Dynamics model | UDP/  | - Flight computer|
//   | - Env model      | Serial| - Sensors (gyro, |
//   | - Sensor models  |       |   accelerometer) |
//   | - Actuator models|       | - Actuators (RW, |
//   +------------------+       |   thrusters)     |
//                              +------------------+
//
//   The simulation sends simulated sensor data to the hardware, and the
//   hardware sends actuator commands back. This closes the loop, allowing
//   the flight software to run on real hardware while the "world" is
//   simulated.
//
//   Benefits:
//   - Test flight software on actual hardware before launch
//   - Find timing issues that pure software simulation misses
//   - Verify electrical interfaces and data formats
//   - Regression testing with repeatable scenarios
//
// COMMUNICATION PROTOCOLS:
//
//   Serial (RS-422/UART): Traditional spacecraft interface. Low latency,
//   deterministic timing. Typical: 115200 baud for housekeeping,
//   1 Mbaud for high-rate sensor data.
//
//   UDP: Used for HIL over Ethernet. Lower overhead than TCP (no handshake,
//   no retransmission). Packet loss is acceptable -- we interpolate.
//   Typical: 1-10 ms update rate, <1 ms latency on a local network.
//
// CRC (CYCLIC REDUNDANCY CHECK):
//
//   Every packet includes a CRC to detect transmission errors. We use
//   CRC-16/CCITT (polynomial 0x1021), which is standard in aerospace:
//   - CCSDS telecommand and telemetry
//   - MIL-STD-1553
//   - SpaceWire
//
//   CRC-16 detects all single-bit errors, all double-bit errors, any odd
//   number of errors, and all burst errors up to 16 bits.
//
// TIMESTAMP SYNCHRONIZATION:
//
//   The simulation and hardware have independent clocks. We must synchronize
//   them to correlate sensor data with the correct simulation state.
//
//   Approach: We use a simplified PTP-like (Precision Time Protocol) scheme:
//   1. Sim sends a "sync" packet with its timestamp T1
//   2. Hardware records receive time T2 and sends response with T2 and T3
//   3. Sim records receive time T4
//   4. One-way delay = ((T4-T1) - (T3-T2)) / 2
//   5. Clock offset = ((T2-T1) + (T3-T4)) / 2
//
//   This gives sub-millisecond synchronization over Ethernet.
//
// =============================================================================

#ifndef GNC_HIL_INTERFACE_H
#define GNC_HIL_INTERFACE_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>

// ---------------------------------------------------------------------------
// Packet structures -- packed for wire transmission
// ---------------------------------------------------------------------------

#pragma pack(push, 1)

// Sync byte that marks the start of every packet
constexpr uint8_t SYNC_BYTE = 0x7E;

// Packet types
enum class PacketType : uint8_t {
    SENSOR_DATA    = 0x01,  // Sensor measurements (hardware -> sim)
    ACTUATOR_CMD   = 0x02,  // Actuator commands (sim -> hardware)
    TIME_SYNC      = 0x03,  // Timestamp synchronization
    HEALTH_STATUS  = 0x04,  // System health telemetry
    CONFIG_CMD     = 0x05,  // Configuration command
    ACK            = 0x06,  // Acknowledgment
    NAK            = 0x07,  // Negative acknowledgment
    HEARTBEAT      = 0x08,  // Keepalive
};

// ---------------------------------------------------------------------------
// Packet header -- common to all packets (8 bytes)
// ---------------------------------------------------------------------------
struct PacketHeader {
    uint8_t  sync;           // Must be SYNC_BYTE (0x7E)
    PacketType type;         // Packet type
    uint16_t sequence;       // Monotonic sequence number (for drop detection)
    uint16_t payload_len;    // Length of payload in bytes (excluding header + CRC)
    uint16_t reserved;       // Reserved for future use (alignment)
};

// ---------------------------------------------------------------------------
// Sensor data packet (hardware -> simulation)
//
// Contains IMU, GPS, and star tracker measurements.
// This is what the real flight hardware would send to the simulation
// during HIL testing.
// ---------------------------------------------------------------------------
struct SensorPacket {
    PacketHeader header;

    // Timestamp (hardware clock, microseconds since boot)
    uint64_t hw_timestamp_us;

    // IMU: Inertial Measurement Unit
    // 3-axis gyroscope (rad/s) + 3-axis accelerometer (m/s^2)
    float gyro[3];          // Angular rate measurement
    float accel[3];         // Specific force measurement
    float gyro_temp;        // Gyro temperature (for bias calibration)

    // GPS: Global Positioning System
    double gps_pos_ecef[3]; // ECEF position (m)
    double gps_vel_ecef[3]; // ECEF velocity (m/s)
    float  gps_pdop;        // Position dilution of precision
    uint8_t gps_num_sats;   // Number of satellites tracked
    uint8_t gps_fix_type;   // 0=no fix, 1=2D, 2=3D, 3=DGPS

    // Star tracker
    float star_quat[4];     // Measured attitude quaternion
    float star_confidence;   // Confidence metric (0-1)
    uint8_t star_num_stars;  // Number of stars identified

    // Sun sensor
    float sun_vector[3];    // Sun direction in body frame (unit vector)
    uint8_t sun_valid;      // 1 if sun is visible

    // CRC-16 of entire packet (header + payload)
    uint16_t crc;
};

// ---------------------------------------------------------------------------
// Actuator command packet (simulation -> hardware)
//
// Commands for reaction wheels, thrusters, and magnetic torquers.
// ---------------------------------------------------------------------------
struct ActuatorCommandPacket {
    PacketHeader header;

    // Timestamp (simulation clock)
    uint64_t sim_timestamp_us;

    // Reaction wheels: 4 wheels in a pyramid configuration
    float rw_torque_cmd[4];    // Commanded torque (N*m)
    float rw_speed_cmd[4];     // Commanded speed (rad/s), or NaN for torque mode

    // Thrusters: 8 thrusters for 6-DOF control
    float thruster_cmd[8];      // Thrust level (0.0 to 1.0)
    uint8_t thruster_enable;    // Bitmask: bit i enables thruster i

    // Magnetic torquers: 3-axis
    float mtq_dipole_cmd[3];   // Commanded magnetic dipole (A*m^2)

    // CRC-16
    uint16_t crc;
};

// ---------------------------------------------------------------------------
// Time synchronization packet
// ---------------------------------------------------------------------------
struct TimeSyncPacket {
    PacketHeader header;
    uint64_t t1;  // Sender's transmit timestamp (us)
    uint64_t t2;  // Receiver's receive timestamp (us) -- filled by receiver
    uint64_t t3;  // Receiver's transmit timestamp (us) -- filled by receiver
    uint16_t crc;
};

// ---------------------------------------------------------------------------
// Health status packet
// ---------------------------------------------------------------------------
struct HealthPacket {
    PacketHeader header;
    uint8_t  mode;              // System mode
    float    cpu_temp;          // CPU temperature (deg C)
    float    bus_voltage;       // Power bus voltage (V)
    float    bus_current;       // Power bus current (A)
    uint32_t uptime_s;          // System uptime (seconds)
    uint32_t error_count;       // Total error count
    uint16_t crc;
};

#pragma pack(pop)

// ---------------------------------------------------------------------------
// Connection configuration
// ---------------------------------------------------------------------------
struct HILConfig {
    enum class Protocol { SERIAL, UDP };

    Protocol protocol = Protocol::UDP;

    // Serial settings
    std::string serial_port = "/dev/ttyUSB0";
    uint32_t    baud_rate   = 115200;

    // UDP settings
    std::string udp_host    = "127.0.0.1";
    uint16_t    udp_port_tx = 5000;  // Send to this port
    uint16_t    udp_port_rx = 5001;  // Receive on this port

    // Timing
    double timeout_ms       = 100.0;   // Receive timeout
    double sync_interval_s  = 1.0;     // Time sync interval
};

// ---------------------------------------------------------------------------
// Connection statistics
// ---------------------------------------------------------------------------
struct HILStats {
    std::size_t packets_sent;
    std::size_t packets_received;
    std::size_t packets_dropped;       // Detected via sequence gaps
    std::size_t crc_errors;
    std::size_t timeout_count;
    double      clock_offset_us;       // Estimated clock offset
    double      one_way_delay_us;      // Estimated one-way latency
    double      avg_round_trip_us;     // Average round-trip time
    double      link_uptime_s;         // Time since connection
    bool        connected;
};

// ---------------------------------------------------------------------------
// Data log entry for post-test analysis
// ---------------------------------------------------------------------------
struct HILLogEntry {
    double timestamp;
    enum class Direction { SEND, RECEIVE } direction;
    PacketType type;
    uint16_t sequence;
    std::size_t size_bytes;
    bool crc_valid;
};

// ---------------------------------------------------------------------------
// HILInterface class
// ---------------------------------------------------------------------------
class HILInterface {
public:
    HILInterface();
    ~HILInterface();

    // No copy
    HILInterface(const HILInterface&) = delete;
    HILInterface& operator=(const HILInterface&) = delete;

    // -----------------------------------------------------------------------
    // Connection management
    // -----------------------------------------------------------------------

    // Connect using specified configuration
    bool connect(const HILConfig& config);

    // Connect with simple parameters (convenience overload)
    bool connect(const std::string& port, uint32_t baud_rate = 115200);

    // Disconnect and clean up
    void disconnect();

    // Check if connected
    bool is_connected() const;

    // -----------------------------------------------------------------------
    // Data exchange
    // -----------------------------------------------------------------------

    // Send actuator commands to hardware
    bool send_actuator_commands(const ActuatorCommandPacket& cmds);

    // Receive sensor data from hardware (blocking with timeout)
    std::optional<SensorPacket> receive_sensor_data();

    // Send health request
    bool send_health_request();

    // Receive health status
    std::optional<HealthPacket> receive_health_status();

    // -----------------------------------------------------------------------
    // Timestamp synchronization
    // -----------------------------------------------------------------------

    // Perform a time sync exchange
    bool perform_time_sync();

    // Get current estimated clock offset
    double get_clock_offset_us() const;

    // Convert hardware timestamp to simulation time
    double hw_to_sim_time(uint64_t hw_timestamp_us) const;

    // -----------------------------------------------------------------------
    // CRC utilities (public for testing)
    // -----------------------------------------------------------------------

    // CRC-16/CCITT (polynomial 0x1021, init 0xFFFF)
    static uint16_t compute_crc16(const uint8_t* data, std::size_t len);

    // Validate a packet's CRC
    static bool validate_packet_crc(const uint8_t* packet, std::size_t len);

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------
    HILStats get_stats() const;
    std::vector<HILLogEntry> get_log() const;
    void clear_log();

    // Set callback for received packets (for async processing)
    void set_receive_callback(std::function<void(const SensorPacket&)> cb);

private:
    // -----------------------------------------------------------------------
    // Internal communication methods
    // -----------------------------------------------------------------------

    // Low-level send/receive (platform-abstracted)
    bool send_raw(const uint8_t* data, std::size_t len);
    int  receive_raw(uint8_t* buffer, std::size_t max_len, double timeout_ms);

    // Packet encoding: add header, compute CRC
    std::vector<uint8_t> encode_packet(PacketType type, const uint8_t* payload,
                                        std::size_t payload_len);

    // Packet decoding: validate sync, check CRC, extract payload
    bool decode_packet(const uint8_t* data, std::size_t len,
                       PacketHeader& header, const uint8_t*& payload);

    // Log a packet event
    void log_event(HILLogEntry::Direction dir, PacketType type,
                   uint16_t seq, std::size_t size, bool crc_ok);

    // -----------------------------------------------------------------------
    // Mock hardware (for testing without real hardware)
    // -----------------------------------------------------------------------
    struct MockHardware {
        bool enabled = false;
        double sim_delay_us = 500.0;   // Simulated communication delay
        double noise_std = 0.001;       // Sensor noise standard deviation
        SensorPacket last_sensor;       // Last generated sensor packet
        uint16_t sequence = 0;

        // Generate fake sensor data based on truth state
        SensorPacket generate_sensor_data(double time_s);
    };

    // -----------------------------------------------------------------------
    // Data members
    // -----------------------------------------------------------------------
    HILConfig    config_;
    HILStats     stats_;
    MockHardware mock_;

    int          socket_fd_;           // Socket file descriptor (for UDP)
    bool         connected_;
    uint16_t     tx_sequence_;         // Transmit sequence counter

    double       clock_offset_us_;     // Estimated clock offset
    double       one_way_delay_us_;    // Estimated one-way delay

    std::vector<HILLogEntry> log_;     // Packet log for post-analysis
    std::function<void(const SensorPacket&)> receive_callback_;

    std::chrono::steady_clock::time_point connect_time_;
};

#endif // GNC_HIL_INTERFACE_H
