package com.group10.fashion.history;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface OutfitHistoryRepository extends JpaRepository<OutfitHistory, Long> {
    List<OutfitHistory> findByUserEmailOrderByCreatedAtDesc(String email);
}
