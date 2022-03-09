export module PyNet.DI:InstanceMode;

export namespace PyNet::DI {

    enum class InstanceMode {
        Unique,
        Shared
    };
}
