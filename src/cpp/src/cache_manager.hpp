

#pragma once

#include <vector>
#include <list>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>
#include <numaif.h>

#include "openvino/runtime/tensor.hpp"

#include "device_config.hpp"

namespace ov::genai {
class CacheManager {
    DeviceConfig m_device_config;
    std::vector<ov::Tensor> m_key_cache;
    std::vector<ov::Tensor> m_value_cache;

public:
    explicit CacheManager(const DeviceConfig& device_config) :
        m_device_config(device_config) {
        m_key_cache.reserve(m_device_config.get_num_layers());
        m_value_cache.reserve(m_device_config.get_num_layers());

        // Allocate KV caches
        for (size_t decoder_layer_id = 0; decoder_layer_id < m_device_config.get_num_layers(); ++decoder_layer_id) {
            ov::Tensor key_cache(device_config.get_cache_precision(), device_config.get_key_cache_shape());
            ov::Tensor value_cache(device_config.get_cache_precision(), device_config.get_value_cache_shape());

            // force allocation
            std::memset(key_cache.data(), 0, key_cache.get_byte_size());
            std::memset(value_cache.data(), 0, value_cache.get_byte_size());

            // force numa aware
            if (std::getenv("KV_NUMA")) {
                // std::cout << "[DEBUG] start KV cache numa realloc ...\n";
                numa_aware_alloc(key_cache);
                numa_aware_alloc(value_cache);
            }

            m_key_cache.emplace_back(key_cache);
            m_value_cache.emplace_back(value_cache);
        }
    }

    bool mem_bind_move(void* data, size_t size, int targetNode, int pagesize, int page_count) {
        // int realNode = ov::get_org_numa_id(targetNode);
        int realNode = targetNode;
        // auto pagesize = getpagesize();
        // auto page_count = (size + pagesize - 1) / pagesize;
        char* pages = reinterpret_cast<char*>((((uintptr_t)data) & ~((uintptr_t)(pagesize - 1))));
        unsigned long mask = 0;
        unsigned flags = 0;
        if (realNode < 0) {
            // restore default policy
            mask = -1;
            flags = 0;
        } else {
            mask = 1ul << realNode;
            flags = MPOL_MF_MOVE | MPOL_MF_STRICT;
        }

        auto rc = mbind(pages, page_count * pagesize, MPOL_BIND, &mask, sizeof(mask) * 8, flags);
        if (rc < 0) {
            std::cout << "mbind failed: " << strerror(errno) << "\n";
            return false;
        }
        return true;
    }

    void numa_aware_alloc(ov::Tensor& cache, int axis = -1) {
        auto data_type = cache.get_element_type();
        auto data_shap = cache.get_shape().to_string();
        auto data_size = cache.get_size();
        auto data_mems = cache.get_byte_size();
        // std::cout << "[debug] tensor type: " << data_type
        //     << ", shape: " << data_shap
        //     << ", size: " << data_size 
        //     << ", data_mems: " << data_mems << "\n";
        // page
        const size_t page_size = getpagesize();
        auto page_nums = (data_mems + page_size - 1) / page_size;
        // std::cout << "[debug] page size: " << page_size << ", page nums: " << page_nums << "\n";
        // dim TODO@alan
        const int head_dim = axis; // should be 1
        auto page_offset = page_nums / 2;
        auto data_ptr = cache.data(data_type);
        // move to node 0
        mem_bind_move(data_ptr, 0, 0, page_size, page_offset);
        // move to node 1
        mem_bind_move(data_ptr, 0, 1, page_size, page_offset);
    }

    ov::Tensor get_key_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_key_cache.size());
        return m_key_cache[decoder_layer_id];
    }

    ov::Tensor get_value_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_value_cache.size());
        return m_value_cache[decoder_layer_id];
    }

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) {
        ov::Shape key_shape = m_device_config.get_key_cache_shape();
        ov::Shape value_shape = m_device_config.get_value_cache_shape();

        ov::Coordinate key_src_start_roi(key_shape.size(), 0);
        ov::Coordinate key_src_end_roi = key_shape;
        ov::Coordinate key_dst_start_roi(key_shape.size(), 0);
        ov::Coordinate key_dst_end_roi = key_shape;
        
        ov::Coordinate value_src_start_roi(value_shape.size(), 0);
        ov::Coordinate value_src_end_roi = value_shape;
        ov::Coordinate value_dst_start_roi(value_shape.size(), 0);
        ov::Coordinate value_dst_end_roi = value_shape;

        for (const auto & blocks_pair : block_copy_map) {
            size_t src_block_id = blocks_pair.first;
            key_src_end_roi[0] = (key_src_start_roi[0] = src_block_id) + 1;
            value_src_end_roi[0] = (value_src_start_roi[0] = src_block_id) + 1;

            const std::list<size_t>& dst_block_ids = blocks_pair.second;
            for (size_t dst_block_id : dst_block_ids) {
                key_dst_end_roi[0] = (key_dst_start_roi[0] = dst_block_id) + 1;
                value_dst_end_roi[0] = (value_dst_start_roi[0] = dst_block_id) + 1;

                for (size_t decoder_layer_id = 0; decoder_layer_id < m_device_config.get_num_layers(); ++decoder_layer_id) {
                    ov::Tensor key_src_cache_roi(m_key_cache[decoder_layer_id], key_src_start_roi, key_src_end_roi);
                    ov::Tensor key_dst_cache_roi(m_key_cache[decoder_layer_id], key_dst_start_roi, key_dst_end_roi);

                    ov::Tensor value_src_cache_roi(m_value_cache[decoder_layer_id], value_src_start_roi, value_src_end_roi);
                    ov::Tensor value_dst_cache_roi(m_value_cache[decoder_layer_id], value_dst_start_roi, value_dst_end_roi);

                    key_src_cache_roi.copy_to(key_dst_cache_roi);
                    value_src_cache_roi.copy_to(value_dst_cache_roi);
                }
            }
        }
    }
};
}
