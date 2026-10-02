package com.group10.fashion.history;

import com.group10.fashion.user.User;
import com.group10.fashion.user.UserRepository;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.time.Instant;
import java.util.List;

@RestController
@RequestMapping("/api/history")
public class HistoryController {

    public record SaveRequest(String title, @NotEmpty List<String> itemIds) {}
    public record HistoryItem(Long id, String title, List<String> itemIds, Instant createdAt) {}

    private final OutfitHistoryRepository repo;
    private final UserRepository users;

    public HistoryController(OutfitHistoryRepository repo, UserRepository users) {
        this.repo = repo;
        this.users = users;
    }

    @GetMapping
    public List<HistoryItem> list(Authentication auth) {
        return repo.findByUserEmailOrderByCreatedAtDesc(auth.getName()).stream()
            .map(h -> new HistoryItem(h.getId(), h.getTitle(),
                h.getItemIds() == null || h.getItemIds().isBlank() ? List.of() : List.of(h.getItemIds().split(",")),
                h.getCreatedAt()))
            .toList();
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public HistoryItem save(Authentication auth, @Valid @RequestBody SaveRequest req) {
        User u = users.findByEmail(auth.getName())
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED));
        OutfitHistory h = repo.save(new OutfitHistory(u, req.title(), String.join(",", req.itemIds())));
        return new HistoryItem(h.getId(), h.getTitle(), req.itemIds(), h.getCreatedAt());
    }
}
