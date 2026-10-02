package com.group10.fashion.user;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "users")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private String email;

    @Column(nullable = false)
    private String passwordHash;

    @Column(nullable = false)
    private String displayName;

    /** Câu trả lời onboarding (style, body shape, occasion) lưu dạng JSON. Làm ở sprint sau. */
    @Column(columnDefinition = "TEXT")
    private String styleProfileJson;

    @Column(nullable = false, updatable = false)
    private Instant createdAt = Instant.now();

    protected User() {}

    public User(String email, String passwordHash, String displayName) {
        this.email = email;
        this.passwordHash = passwordHash;
        this.displayName = displayName;
    }

    public Long getId() { return id; }
    public String getEmail() { return email; }
    public String getPasswordHash() { return passwordHash; }
    public String getDisplayName() { return displayName; }
    public String getStyleProfileJson() { return styleProfileJson; }
    public void setStyleProfileJson(String styleProfileJson) { this.styleProfileJson = styleProfileJson; }
    public Instant getCreatedAt() { return createdAt; }
}
