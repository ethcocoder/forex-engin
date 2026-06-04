#include <atomic>
#include <vector>
#include <cstdint>
#include <cstring>

/**
 * Lock-free SPSC Ring Buffer for Ultra-Low Latency Tick Ingestion.
 * 
 * Uses memory order release/acquire semantics and cache line padding (64 bytes)
 * to eliminate CPU cache false sharing, providing maximum performance on multi-core systems.
 */

struct TickData {
    char pair[16];
    double price;
    double bid;
    double ask;
    double volume;
    double timestamp;
};

class SPSCRingBuffer {
public:
    explicit SPSCRingBuffer(size_t capacity)
        : capacity_(capacity),
          buffer_(capacity + 1),
          head_(0),
          tail_(0) {}

    bool enqueue(const TickData& tick) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_acquire);
        
        size_t next_head = (head + 1) % buffer_.size();
        if (next_head == tail) {
            return false; // Queue full
        }
        buffer_[head] = tick;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    bool dequeue(TickData& tick) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t head = head_.load(std::memory_order_acquire);
        
        if (tail == head) {
            return false; // Queue empty
        }
        tick = buffer_[tail];
        tail_.store((tail + 1) % buffer_.size(), std::memory_order_release);
        return true;
    }

    size_t size() const {
        size_t head = head_.load(std::memory_order_acquire);
        size_t tail = tail_.load(std::memory_order_acquire);
        if (head >= tail) {
            return head - tail;
        } else {
            return buffer_.size() - tail + head;
        }
    }

private:
    size_t capacity_;
    std::vector<TickData> buffer_;
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
};

extern "C" {

SPSCRingBuffer* create_spsc_buffer(int capacity) {
    return new SPSCRingBuffer(capacity);
}

void destroy_spsc_buffer(SPSCRingBuffer* buffer) {
    delete buffer;
}

bool spsc_enqueue(SPSCRingBuffer* buffer, const TickData* tick) {
    if (!buffer || !tick) return false;
    return buffer->enqueue(*tick);
}

bool spsc_dequeue(SPSCRingBuffer* buffer, TickData* tick) {
    if (!buffer || !tick) return false;
    return buffer->dequeue(*tick);
}

int spsc_size(SPSCRingBuffer* buffer) {
    if (!buffer) return 0;
    return static_cast<int>(buffer->size());
}

}
