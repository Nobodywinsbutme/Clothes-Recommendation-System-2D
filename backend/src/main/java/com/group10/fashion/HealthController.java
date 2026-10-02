package com.group10.fashion;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClient;

import java.util.Map;

@RestController
public class HealthController {

    private final RestClient aiClient;

    public HealthController(@Value("${app.ai-service.url}") String aiUrl) {
        this.aiClient = RestClient.create(aiUrl);
    }

    @GetMapping("/api/health")
    public Map<String, String> health() {
        return Map.of("service", "backend", "status", "ok");
    }

    /** Kiểm tra backend gọi được sang ai-service (hello world liên service). */
    @GetMapping("/api/ping-ai")
    public Map<String, Object> pingAi() {
        try {
            String body = aiClient.get().uri("/health").retrieve().body(String.class);
            return Map.of("backend", "ok", "aiService", body);
        } catch (Exception e) {
            return Map.of("backend", "ok", "aiService", "unreachable: " + e.getMessage());
        }
    }
}
