package com.group10.fashion.history;

import com.group10.fashion.user.User;
import jakarta.persistence.*;
import java.time.Instant;

/** Một lần phối đồ của người dùng (lưu danh sách id món đồ DeepFashion đã chọn/like). */
@Entity
@Table(name = "outfit_history")
public class OutfitHistory {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id")
    private User user;

    private String title;

    /** id các món đồ, cách nhau bởi dấu phẩy. */
    @Column(columnDefinition = "TEXT")
    private String itemIds;

    @Column(nullable = false, updatable = false)
    private Instant createdAt = Instant.now();

    protected OutfitHistory() {}

    public OutfitHistory(User user, String title, String itemIds) {
        this.user = user;
        this.title = title;
        this.itemIds = itemIds;
    }

    public Long getId() { return id; }
    public String getTitle() { return title; }
    public String getItemIds() { return itemIds; }
    public Instant getCreatedAt() { return createdAt; }
}
